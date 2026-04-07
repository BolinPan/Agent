#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Document Q&A Assistant - Intelligent Document Q&A System Based on HelloAgents

This is a complete PDF learning assistant application that supports:
- Loading PDF documents and building knowledge base
- Intelligent Q&A (based on RAG)
- Learning history recording (based on Memory)
- Learning review and report generation
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from hello_agents.tools import MemoryTool, RAGTool
import gradio as gr


class PDFLearningAssistant:
    """
    Intelligent Document Q&A Assistant
    """

    def __init__(self, user_id: str = "default_user"):
        """
        Initialize learning assistant

        Args:
            user_id: User ID for isolating data of different users
        """
        self.user_id = user_id
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize tools
        self.memory_tool = MemoryTool(user_id=user_id)
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # Learning statistics
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0
        }

        # Currently loaded document
        self.current_document = None


    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        Load PDF document into knowledge base

        Args:
            pdf_path: PDF file path

        Returns:
            Dict: Result containing success and message
        """
        if not os.path.exists(pdf_path):
            return {"success": False, "message": f"File does not exist: {pdf_path}"}

        start_time = time.time()

        try:
            # Use RAG tool to process PDF
            result = self.rag_tool.execute(
                "add_document",
                file_path=pdf_path,
                chunk_size=1000,
                chunk_overlap=200
            )

            process_time = time.time() - start_time

            # RAG tool returns a string message
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            # Record to learning memory
            self.memory_tool.execute(
                "add",
                content=f"Loaded document '{self.current_document}'",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id
            )

            return {
                "success": True,
                "message": f"Loading successful! (Time taken: {process_time:.1f}s)",
                "document": self.current_document
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Loading failed: {str(e)}"
            }

    def ask(self, question: str, use_advanced_search: bool = True) -> str:
        """Ask questions about the document

        Args:
            question: User question
            use_advanced_search: Whether to use advanced search (MQE + HyDE)

        Returns:
            str: Answer
        """
        if not self.current_document:
            return "⚠️ Please load a document first! Use the load_document() method to load a PDF document."

        # Record question to working memory
        self.memory_tool.execute(
            "add",
            content=f"Question: {question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id
        )

        # Use RAG to retrieve answer
        answer = self.rag_tool.execute(
            "ask",
            question=question,
            limit=5,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search
        )

        # Record to episodic memory
        self.memory_tool.execute(
            "add",
            content=f"Learning about '{question}'",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id
        )

        self.stats["questions_asked"] += 1

        return answer

    def add_note(self, content: str, concept: Optional[str] = None):
        """Add learning notes

        Args:
            content: Note content
            concept: Related concept (optional)
        """
        self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id
        )

        self.stats["concepts_learned"] += 1

    def recall(self, query: str, limit: int = 5) -> str:
        """Review learning history

        Args:
            query: Query keywords
            limit: Number of results to return

        Returns:
            str: Related memories
        """
        result = self.memory_tool.execute(
            "search",
            query=query,
            limit=limit
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics

        Returns:
            Dict: Statistics information
        """
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()

        return {
            "Session Duration": f"{duration:.0f}s",
            "Documents Loaded": self.stats["documents_loaded"],
            "Questions Asked": self.stats["questions_asked"],
            "Learning Notes": self.stats["concepts_learned"],
            "Current Document": self.current_document or "Not loaded"
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate learning report

        Args:
            save_to_file: Whether to save to file

        Returns:
            Dict: Learning report
        """
        # Get memory summary
        memory_summary = self.memory_tool.execute("summary", limit=10)

        # Get RAG statistics
        rag_stats = self.rag_tool.execute("stats")

        # Generate report
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"]
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats
        }

        # Save to file
        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2, default=str)
                report["report_file"] = report_file
            except Exception as e:
                report["save_error"] = str(e)

        return report





def create_gradio_ui():
    """Create Gradio Web UI"""
    # Global assistant instance
    assistant_state = {"assistant": None}

    def init_assistant(user_id: str) -> str:
        """Initialize assistant"""
        if not user_id:
            user_id = "web_user"
        assistant_state["assistant"] = PDFLearningAssistant(user_id=user_id)
        return f"✅ Assistant initialized (User: {user_id})"

    def load_pdf(pdf_file) -> str:
        """Load PDF file"""
        if assistant_state["assistant"] is None:
            return "❌ Please initialize assistant first"

        if pdf_file is None:
            return "❌ Please upload a PDF file"

        # Gradio uploaded file is a temporary file object
        pdf_path = pdf_file.name
        result = assistant_state["assistant"].load_document(pdf_path)

        if result["success"]:
            return f"✅ {result['message']}\n📄 Document: {result['document']}"
        else:
            return f"❌ {result['message']}"

    def chat(message: str, history: List) -> Tuple[str, List]:
        """Chat functionality"""
        if assistant_state["assistant"] is None:
            return "", history + [[message, "❌ Please initialize assistant and load document first"]]

        if not message.strip():
            return "", history

        # Determine if it's a technical question or review question
        if any(keyword in message for keyword in ["before", "learned", "review", "history", "remember"]):
            # Review learning history
            response = assistant_state["assistant"].recall(message)
            response = f"🧠 **Learning Review**\n\n{response}"
        else:
            # Technical Q&A
            response = assistant_state["assistant"].ask(message)
            response = f"💡 **Answer**\n\n{response}"

        history.append([message, response])
        return "", history

    def add_note_ui(note_content: str, concept: str) -> str:
        """Add notes"""
        if assistant_state["assistant"] is None:
            return "❌ Please initialize assistant first"

        if not note_content.strip():
            return "❌ Note content cannot be empty"

        assistant_state["assistant"].add_note(note_content, concept or None)
        return f"✅ Note saved: {note_content[:50]}..."

    def get_stats_ui() -> str:
        """Get statistics information"""
        if assistant_state["assistant"] is None:
            return "❌ Please initialize assistant first"

        stats = assistant_state["assistant"].get_stats()
        result = "📊 **Learning Statistics**\n\n"
        for key, value in stats.items():
            result += f"- **{key}**: {value}\n"
        return result

    def generate_report_ui() -> str:
        """Generate report"""
        if assistant_state["assistant"] is None:
            return "❌ Please initialize assistant first"

        report = assistant_state["assistant"].generate_report(save_to_file=True)

        result = f"✅ Learning report generated\n\n"
        result += f"**Session Information**\n"
        result += f"- Session Duration: {report['session_info']['duration_seconds']:.0f}s\n"
        result += f"- Documents Loaded: {report['learning_metrics']['documents_loaded']}\n"
        result += f"- Questions Asked: {report['learning_metrics']['questions_asked']}\n"
        result += f"- Learning Notes: {report['learning_metrics']['concepts_learned']}\n"

        if "report_file" in report:
            result += f"\n💾 Report saved to: {report['report_file']}"

        return result

    # Create Gradio interface
    with gr.Blocks(title="Intelligent Document Q&A Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📚 Intelligent Document Q&A Assistant

        Intelligent document Q&A system based on HelloAgents, supporting:
        - 📄 Loading PDF documents and building knowledge base
        - 💬 Intelligent Q&A (based on RAG)
        - 📝 Learning notes recording
        - 🧠 Learning history review
        - 📊 Learning report generation
        """)

        with gr.Tab("🏠 Getting Started"):
            with gr.Row():
                user_id_input = gr.Textbox(
                    label="User ID",
                    placeholder="Enter your user ID (optional, defaults to web_user)",
                    value="web_user"
                )
                init_btn = gr.Button("Initialize Assistant", variant="primary")

            init_output = gr.Textbox(label="Initialization Status", interactive=False)
            init_btn.click(init_assistant, inputs=[user_id_input], outputs=[init_output])

            gr.Markdown("### 📄 Load PDF Document")
            pdf_upload = gr.File(
                label="Upload PDF File",
                file_types=[".pdf"],
                type="filepath"
            )
            load_btn = gr.Button("Load Document", variant="primary")
            load_output = gr.Textbox(label="Loading Status", interactive=False)
            load_btn.click(load_pdf, inputs=[pdf_upload], outputs=[load_output])

        with gr.Tab("💬 Intelligent Q&A"):
            gr.Markdown("### Ask questions about the document or review learning history")
            chatbot = gr.Chatbot(
                label="Chat History",
                height=400,
                bubble_full_width=False
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Enter Question",
                    placeholder="For example: What is Transformer? or What have I learned before?",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            gr.Examples(
                examples=[
                    "What is a large language model?",
                    "What are the core components of Transformer architecture?",
                    "How to train large language models?",
                    "What content have I learned before?",
                    "Review the learning about attention mechanism"
                ],
                inputs=msg_input
            )

            msg_input.submit(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            send_btn.click(chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])

        with gr.Tab("📝 Learning Notes"):
            gr.Markdown("### Record learning insights and important concepts")
            note_content = gr.Textbox(
                label="Note Content",
                placeholder="Enter your learning notes...",
                lines=3
            )
            concept_input = gr.Textbox(
                label="Related Concept (Optional)",
                placeholder="For example: transformer, attention"
            )
            note_btn = gr.Button("Save Note", variant="primary")
            note_output = gr.Textbox(label="Save Status", interactive=False)
            note_btn.click(add_note_ui, inputs=[note_content, concept_input], outputs=[note_output])

        with gr.Tab("📊 Learning Statistics"):
            gr.Markdown("### View learning progress and statistics")
            stats_btn = gr.Button("Refresh Statistics", variant="primary")
            stats_output = gr.Markdown()
            stats_btn.click(get_stats_ui, outputs=[stats_output])

            gr.Markdown("### Generate Learning Report")
            report_btn = gr.Button("Generate Report", variant="primary")
            report_output = gr.Textbox(label="Report Status", interactive=False)
            report_btn.click(generate_report_ui, outputs=[report_output])

    return demo


def main():
    """Main function - Launch Gradio Web UI"""
    print("\n" + "="*60)
    print("🧠 Intelligent Document Q&A Assistant")
    print("="*60)
    print("Launching web interface...\n")

    demo = create_gradio_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()

