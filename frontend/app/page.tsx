"use client";

import { useState } from "react";
import {
  Send,
  Upload,
  FileText,
  Trash2,
  Plus,
  Bot,
  User,
  Loader2,
} from "lucide-react";

const API_URL = "http://localhost:8000";

interface Source {
  document_id: string;
  chunk_id: string;
  page_number: number;
  score: number;
  text: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
}

interface Document {
  document_id: string;
  filename: string | null;
  pages: number;
  chunks: number;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);

  // ---------------------------------------------------------
  // SEND QUESTION
  // ---------------------------------------------------------

  async function sendQuestion() {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || loading) return;

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: trimmedQuestion,
      },
    ]);

    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}/api/v1/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: trimmedQuestion,
            limit: 5,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `API error: ${response.status}`
        );
      }

      const data = await response.json();

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
        },
      ]);
    } catch (error) {
      console.error(error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "I couldn't connect to the OmniBrain backend. Make sure FastAPI is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // ---------------------------------------------------------
  // HANDLE ENTER
  // ---------------------------------------------------------

  function handleKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();
      sendQuestion();
    }
  }

  // ---------------------------------------------------------
  // UPLOAD DOCUMENT
  // ---------------------------------------------------------

  async function uploadDocument(
    event: React.ChangeEvent<HTMLInputElement>
  ) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF files are supported.");
      return;
    }

    setUploading(true);

    const formData = new FormData();

    formData.append("file", file);

    try {
      const response = await fetch(
        `${API_URL}/api/v1/upload`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Upload failed: ${response.status}`
        );
      }

      const data = await response.json();

      setDocuments((previous) => [
        ...previous,
        {
          document_id: data.document_id,
          filename: data.filename,
          pages: data.pages,
          chunks: data.text_chunks,
        },
      ]);

      alert(
        `${data.filename} uploaded and indexed successfully.`
      );
    } catch (error) {
      console.error(error);

      alert(
        "Document upload failed. Make sure the backend is running."
      );
    } finally {
      setUploading(false);

      event.target.value = "";
    }
  }

  // ---------------------------------------------------------
  // DELETE DOCUMENT
  // ---------------------------------------------------------

  async function deleteDocument(
    documentId: string
  ) {
    try {
      const response = await fetch(
        `${API_URL}/api/v1/documents/${documentId}`,
        {
          method: "DELETE",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Delete failed: ${response.status}`
        );
      }

      setDocuments((previous) =>
        previous.filter(
          (document) =>
            document.document_id !== documentId
        )
      );
    } catch (error) {
      console.error(error);

      alert("Could not delete document.");
    }
  }

  // ---------------------------------------------------------
  // NEW CHAT
  // ---------------------------------------------------------

  function newChat() {
    setMessages([]);
  }

  return (
    <main className="min-h-screen bg-[#08090d] text-white flex">
      
      {/* =====================================================
          SIDEBAR
      ===================================================== */}

      <aside className="w-[280px] border-r border-white/10 bg-[#0c0d12] flex flex-col">

        {/* Logo */}

        <div className="p-6 border-b border-white/10">

          <div className="flex items-center gap-3">

            <div className="w-10 h-10 rounded-xl bg-white text-black flex items-center justify-center">
              <Bot size={22} />
            </div>

            <div>
              <h1 className="font-semibold text-lg">
                OmniBrain
              </h1>

              <p className="text-xs text-white/40">
                AI Knowledge Engine
              </p>
            </div>

          </div>

        </div>


        {/* New Chat */}

        <div className="p-4">

          <button
            onClick={newChat}
            className="w-full flex items-center justify-center gap-2 rounded-xl bg-white text-black py-3 font-medium hover:bg-white/90 transition"
          >
            <Plus size={18} />

            New Chat
          </button>

        </div>


        {/* Documents */}

        <div className="px-4 flex-1 overflow-y-auto">

          <div className="flex items-center justify-between mb-3">

            <p className="text-xs uppercase tracking-wider text-white/40">
              Documents
            </p>

            <span className="text-xs text-white/30">
              {documents.length}
            </span>

          </div>


          {/* Upload */}

          <label className="cursor-pointer block">

            <div className="border border-dashed border-white/20 rounded-xl p-4 hover:border-white/40 transition">

              <div className="flex items-center gap-3">

                <div className="w-9 h-9 rounded-lg bg-white/5 flex items-center justify-center">
                  {uploading ? (
                    <Loader2
                      size={18}
                      className="animate-spin"
                    />
                  ) : (
                    <Upload size={18} />
                  )}
                </div>

                <div>

                  <p className="text-sm font-medium">
                    {uploading
                      ? "Processing..."
                      : "Upload PDF"}
                  </p>

                  <p className="text-xs text-white/40">
                    Max 20 MB
                  </p>

                </div>

              </div>

            </div>

            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={uploadDocument}
              disabled={uploading}
            />

          </label>


          {/* Document List */}

          <div className="mt-5 space-y-2">

            {documents.map((document) => (

              <div
                key={document.document_id}
                className="group rounded-xl bg-white/[0.03] border border-white/5 p-3"
              >

                <div className="flex items-start gap-3">

                  <FileText
                    size={18}
                    className="mt-0.5 text-white/50 shrink-0"
                  />

                  <div className="min-w-0 flex-1">

                    <p className="text-sm truncate">
                      {document.filename}
                    </p>

                    <p className="text-xs text-white/30 mt-1">
                      {document.pages} pages ·{" "}
                      {document.chunks} chunks
                    </p>

                  </div>

                  <button
                    onClick={() =>
                      deleteDocument(
                        document.document_id
                      )
                    }
                    className="opacity-0 group-hover:opacity-100 text-white/30 hover:text-red-400 transition"
                  >
                    <Trash2 size={15} />
                  </button>

                </div>

              </div>

            ))}

          </div>

        </div>


        {/* Footer */}

        <div className="p-4 border-t border-white/10">

          <p className="text-xs text-white/30">
            OmniBrain AI v1.0
          </p>

        </div>

      </aside>


      {/* =====================================================
          MAIN CHAT
      ===================================================== */}

      <section className="flex-1 flex flex-col">


        {/* Header */}

        <header className="h-[72px] border-b border-white/10 flex items-center px-8">

          <div>

            <h2 className="font-medium">
              Document Intelligence
            </h2>

            <p className="text-xs text-white/30">
              Ask questions about your documents
            </p>

          </div>

        </header>


        {/* Messages */}

        <div className="flex-1 overflow-y-auto">

          {messages.length === 0 ? (

            <div className="h-full flex items-center justify-center">

              <div className="text-center max-w-lg">

                <div className="mx-auto w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-6">

                  <Bot size={30} />

                </div>

                <h2 className="text-2xl font-semibold mb-3">
                  Ask OmniBrain
                </h2>

                <p className="text-white/40 text-sm leading-relaxed">
                  Upload a PDF and ask questions about
                  its contents. OmniBrain retrieves the
                  most relevant information and generates
                  an answer using your local AI model.
                </p>

              </div>

            </div>

          ) : (

            <div className="max-w-4xl mx-auto px-8 py-8 space-y-8">

              {messages.map((message, index) => (

                <div
                  key={index}
                  className="flex gap-4"
                >

                  {/* Avatar */}

                  <div
                    className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 ${
                      message.role === "user"
                        ? "bg-white/10"
                        : "bg-white text-black"
                    }`}
                  >

                    {message.role === "user" ? (
                      <User size={17} />
                    ) : (
                      <Bot size={17} />
                    )}

                  </div>


                  {/* Content */}

                  <div className="flex-1">

                    <p className="text-xs uppercase tracking-wider text-white/30 mb-2">
                      {message.role === "user"
                        ? "You"
                        : "OmniBrain"}
                    </p>

                    <div className="text-sm leading-7 text-white/85 whitespace-pre-wrap">
                      {message.content}
                    </div>


                    {/* Sources */}

                    {message.sources &&
                      message.sources.length > 0 && (

                        <div className="mt-5">

                          <p className="text-xs uppercase tracking-wider text-white/30 mb-3">
                            Sources
                          </p>

                          <div className="space-y-2">

                            {message.sources.map(
                              (source, sourceIndex) => (

                                <div
                                  key={sourceIndex}
                                  className="rounded-xl border border-white/10 bg-white/[0.02] p-4"
                                >

                                  <div className="flex items-center gap-2 mb-2">

                                    <FileText
                                      size={15}
                                      className="text-white/40"
                                    />

                                    <span className="text-xs text-white/60">
                                      Page{" "}
                                      {
                                        source.page_number
                                      }
                                    </span>

                                    <span className="text-xs text-white/25">
                                      ·
                                    </span>

                                    <span className="text-xs text-white/40">
                                      Relevance{" "}
                                      {(
                                        source.score *
                                        100
                                      ).toFixed(1)}
                                      %
                                    </span>

                                  </div>

                                  <p className="text-xs text-white/40 leading-5 line-clamp-3">
                                    {source.text}
                                  </p>

                                </div>

                              )
                            )}

                          </div>

                        </div>

                      )}

                  </div>

                </div>

              ))}


              {/* Loading */}

              {loading && (

                <div className="flex gap-4">

                  <div className="w-9 h-9 rounded-xl bg-white text-black flex items-center justify-center">

                    <Bot size={17} />

                  </div>

                  <div>

                    <p className="text-xs uppercase tracking-wider text-white/30 mb-2">
                      OmniBrain
                    </p>

                    <div className="flex items-center gap-2 text-white/40 text-sm">

                      <Loader2
                        size={16}
                        className="animate-spin"
                      />

                      Thinking...

                    </div>

                  </div>

                </div>

              )}

            </div>

          )}

        </div>


        {/* Input */}

        <div className="border-t border-white/10 p-5">

          <div className="max-w-4xl mx-auto">

            <div className="relative border border-white/10 rounded-2xl bg-white/[0.03] focus-within:border-white/25 transition">

              <textarea
                value={question}
                onChange={(event) =>
                  setQuestion(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents..."
                rows={1}
                className="w-full bg-transparent resize-none outline-none px-5 py-4 pr-14 text-sm placeholder:text-white/25"
              />

              <button
                onClick={sendQuestion}
                disabled={
                  loading ||
                  !question.trim()
                }
                className="absolute right-3 bottom-2.5 w-9 h-9 rounded-xl bg-white text-black flex items-center justify-center disabled:opacity-20 hover:bg-white/90 transition"
              >

                <Send size={17} />

              </button>

            </div>

            <p className="text-center text-[11px] text-white/20 mt-3">
              OmniBrain can make mistakes. Verify important information.
            </p>

          </div>

        </div>

      </section>

    </main>
  );
}