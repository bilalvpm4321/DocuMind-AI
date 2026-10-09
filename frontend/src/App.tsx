import { useRef, useState } from 'react'
import {
  FileText, Upload, Send, LoaderCircle, Bot, UserRound,
  FileUp, Sparkles, Trash2, Cpu, Cloud, Layers, Settings, X
} from 'lucide-react'

type Source = { page: number | null; kind: string; text: string; score: number }
type Message = {
  role: 'user' | 'assistant'
  content: string
  provider?: 'gemini' | 'ollama' | 'both'
  answers?: { gemini?: string; ollama?: string }
  sources?: Source[]
}
type UploadedDocument = {
  document_id: string
  filename: string
  pages_with_text: number
  chunks: number
  preview: string
}

type ProviderChoice = 'gemini' | 'ollama' | 'both'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function App() {
  const fileInput = useRef<HTMLInputElement>(null)
  const [document, setDocument] = useState<UploadedDocument | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [uploading, setUploading] = useState(false)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [provider, setProvider] = useState<ProviderChoice>('gemini')
  const [showSettings, setShowSettings] = useState(false)

  async function uploadFile(file?: File) {
    if (!file) return
    setError('')
    setUploading(true)
    setDocument(null)
    setMessages([])
    const body = new FormData()
    body.append('file', file)
    try {
      const response = await fetch(`${API_BASE}/api/documents/upload`, { method: 'POST', body })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Upload failed')
      setDocument(data)
      setMessages([{
        role: 'assistant',
        content: `I've processed **${data.filename}** and indexed ${data.chunks} text chunks. You can now ask questions using Gemini, Ollama, or run both to compare answers side-by-side!`
      }])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not upload file')
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function sendQuestion(text = question) {
    const clean = text.trim()
    if (!clean || !document || sending) return
    const prior = messages
    setMessages([...prior, { role: 'user', content: clean }])
    setQuestion('')
    setSending(true)
    setError('')

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          document_id: document.document_id,
          question: clean,
          provider: provider,
          history: prior.map(m => ({ role: m.role, content: m.content })),
        }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Question failed')
      setMessages(current => [
        ...current,
        {
          role: 'assistant',
          content: data.answer,
          provider: data.provider || provider,
          answers: data.answers,
          sources: data.sources
        }
      ])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not get an answer')
      setMessages(current => [
        ...current,
        {
          role: 'assistant',
          content: 'I could not answer that question. Please verify your selected provider settings and backend connection.',
          provider: provider
        }
      ])
    } finally {
      setSending(false)
    }
  }

  function reset() {
    setDocument(null)
    setMessages([])
    setError('')
    setQuestion('')
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Sparkles size={20} /></div>
          <div>
            <strong>DocuMind AI</strong>
            <span>Document Intelligence</span>
          </div>
        </div>

        <button className="new-doc" onClick={() => fileInput.current?.click()}>
          <Upload size={17} /> Upload a document
        </button>
        <input
          ref={fileInput}
          type="file"
          hidden
          accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.webp"
          onChange={e => uploadFile(e.target.files?.[0])}
        />

        <div className="side-label">AI ENGINE SELECTION</div>
        <div className="provider-selector-sidebar">
          <button
            className={`provider-tab ${provider === 'gemini' ? 'active' : ''}`}
            onClick={() => setProvider('gemini')}
          >
            <Cloud size={15} />
            <div className="provider-text">
              <strong>Google Gemini</strong>
              <small>Cloud Fast & Smart</small>
            </div>
          </button>

          <button
            className={`provider-tab ${provider === 'ollama' ? 'active' : ''}`}
            onClick={() => setProvider('ollama')}
          >
            <Cpu size={15} />
            <div className="provider-text">
              <strong>Local Ollama</strong>
              <small>Private & Offline</small>
            </div>
          </button>

          <button
            className={`provider-tab ${provider === 'both' ? 'active' : ''}`}
            onClick={() => setProvider('both')}
          >
            <Layers size={15} />
            <div className="provider-text">
              <strong>Run Both</strong>
              <small>Compare Side-by-Side</small>
            </div>
          </button>
        </div>

        <div className="side-label">WORKSPACE</div>
        {document ? (
          <div className="document-row">
            <FileText size={19} />
            <div className="document-name">
              <strong>{document.filename}</strong>
              <span>{document.chunks} indexed chunks</span>
            </div>
            <button className="icon-button" title="Remove document" onClick={reset}>
              <Trash2 size={15} />
            </button>
          </div>
        ) : (
          <div className="empty-sidebar">Your uploaded documents will appear here.</div>
        )}

        <div className="sidebar-bottom">
          <button className="settings-trigger" onClick={() => setShowSettings(true)}>
            <Settings size={14} /> Provider Settings
          </button>
          <div className="version">v0.2 Multi-LLM</div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="main-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">DOCUMENT RAG INTELLIGENCE</span>
            <h1>Chat with your document</h1>
          </div>

          <div className="topbar-controls">
            {/* Quick Provider Toggle */}
            <div className="pill-group">
              <button
                className={`pill-btn ${provider === 'gemini' ? 'active' : ''}`}
                onClick={() => setProvider('gemini')}
                title="Use Google Gemini API"
              >
                <Cloud size={13} /> Gemini
              </button>
              <button
                className={`pill-btn ${provider === 'ollama' ? 'active' : ''}`}
                onClick={() => setProvider('ollama')}
                title="Use Local Ollama"
              >
                <Cpu size={13} /> Ollama
              </button>
              <button
                className={`pill-btn ${provider === 'both' ? 'active' : ''}`}
                onClick={() => setProvider('both')}
                title="Run Gemini & Ollama simultaneously"
              >
                <Layers size={13} /> Both
              </button>
            </div>

            <div className="model-pill">
              <span className="status-dot" />
              {provider === 'gemini' && 'Gemini 3.8 Flash'}
              {provider === 'ollama' && 'Ollama (Local)'}
              {provider === 'both' && 'Dual Comparison Mode'}
            </div>
          </div>
        </header>

        {!document ? (
          <section className="welcome-area">
            <div className="welcome-icon"><FileUp size={28} /></div>
            <h2>What would you like to explore?</h2>
            <p>
              Upload any PDF, Word document, TXT, or scanned image to ask questions with citations.
              Switch seamlessly between <strong>Google Gemini</strong>, <strong>Local Ollama</strong>, or <strong>Both</strong>.
            </p>

            <button
              className="primary-button"
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
            >
              {uploading ? <LoaderCircle className="spin" size={18} /> : <Upload size={18} />}
              {uploading ? 'Processing & indexing document…' : 'Choose a document'}
            </button>

            <div className="file-types">
              PDF · DOCX · TXT · PNG · JPG · WEBP <span>Up to 20 MB</span>
            </div>

            <div className="suggestion-grid">
              <button onClick={() => fileInput.current?.click()}>
                <FileText size={18} />
                <span>
                  <strong>Financial & Legal Reports</strong>
                  <small>Extract clauses and verify numbers</small>
                </span>
              </button>
              <button onClick={() => fileInput.current?.click()}>
                <Sparkles size={18} />
                <span>
                  <strong>Research & Manuals</strong>
                  <small>Summarize key insights and citations</small>
                </span>
              </button>
            </div>
          </section>
        ) : (
          <>
            <section className="document-summary">
              <div className="file-icon"><FileText size={22} /></div>
              <div className="summary-copy">
                <strong>{document.filename}</strong>
                <span>{document.pages_with_text} content sections · {document.chunks} chunks indexed</span>
              </div>
              <div className="mode-indicator">
                Mode: <strong>{provider.toUpperCase()}</strong>
              </div>
            </section>

            <section className="chat-area" aria-live="polite">
              {messages.map((message, i) => (
                <article className={`message ${message.role}`} key={i}>
                  <div className="avatar">
                    {message.role === 'assistant' ? <Bot size={18} /> : <UserRound size={18} />}
                  </div>

                  <div className="message-body">
                    <div className="message-header-row">
                      <span className="message-label">
                        {message.role === 'assistant' ? 'DOCUMENT AI' : 'YOU'}
                      </span>
                      {message.role === 'assistant' && message.provider && (
                        <span className={`provider-badge ${message.provider}`}>
                          {message.provider === 'gemini' && '🌟 Gemini Flash'}
                          {message.provider === 'ollama' && '🦙 Ollama Local'}
                          {message.provider === 'both' && '⚡ Gemini & Ollama Comparison'}
                        </span>
                      )}
                    </div>

                    {/* If both answers are present, show side-by-side comparison */}
                    {message.answers && message.answers.gemini && message.answers.ollama ? (
                      <div className="comparison-container">
                        <div className="comparison-card gemini-card">
                          <div className="card-header">
                            <Cloud size={15} /> <strong>Google Gemini</strong>
                          </div>
                          <div className="card-content">
                            {message.answers.gemini.split(/\n/).map((line, idx) => (
                              <p key={idx}>{line.replace(/\*\*/g, '')}</p>
                            ))}
                          </div>
                        </div>

                        <div className="comparison-card ollama-card">
                          <div className="card-header">
                            <Cpu size={15} /> <strong>Ollama Local</strong>
                          </div>
                          <div className="card-content">
                            {message.answers.ollama.split(/\n/).map((line, idx) => (
                              <p key={idx}>{line.replace(/\*\*/g, '')}</p>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="message-content">
                        {message.content.split(/\n/).map((line, lineIndex) => (
                          <p key={lineIndex}>{line.replace(/\*\*/g, '')}</p>
                        ))}
                      </div>
                    )}

                    {message.sources && message.sources.length > 0 && (
                      <details className="sources">
                        <summary>Sources retrieved ({message.sources.length})</summary>
                        {message.sources.map((source, j) => (
                          <div className="source-card" key={j}>
                            <strong>{source.page ? `Page ${source.page}` : 'Document excerpt'} (Relevance: {source.score})</strong>
                            <span>{source.text}</span>
                          </div>
                        ))}
                      </details>
                    )}
                  </div>
                </article>
              ))}

              {sending && (
                <div className="thinking">
                  <LoaderCircle size={16} className="spin" />
                  {provider === 'both'
                    ? 'Querying both Gemini and Ollama in parallel…'
                    : `Searching document and asking ${provider === 'gemini' ? 'Gemini' : 'Ollama'}…`}
                </div>
              )}
            </section>

            <div className="chat-composer">
              <form onSubmit={e => { e.preventDefault(); void sendQuestion() }}>
                <textarea
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  placeholder={`Ask a question with ${provider === 'both' ? 'Gemini & Ollama' : provider === 'gemini' ? 'Gemini' : 'Ollama'}…`}
                  rows={2}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      void sendQuestion()
                    }
                  }}
                />
                <div className="composer-footer">
                  <div className="composer-provider-hint">
                    Current engine: <strong>{provider.toUpperCase()}</strong>
                  </div>
                  <button className="send-button" type="submit" disabled={!question.trim() || sending}>
                    <Send size={16} /> Send
                  </button>
                </div>
              </form>
            </div>
          </>
        )}

        {error && <div className="error-banner" role="alert">{error}</div>}

        {/* Settings Modal */}
        {showSettings && (
          <div className="modal-overlay" onClick={() => setShowSettings(false)}>
            <div className="modal-box" onClick={e => e.stopPropagation()}>
              <div className="modal-header">
                <h3><Settings size={18} /> Model & Engine Configuration</h3>
                <button className="icon-button" onClick={() => setShowSettings(false)}><X size={18} /></button>
              </div>

              <div className="modal-body">
                <p>Configure which LLM engines are used for Document Q&A:</p>

                <div className="settings-option">
                  <label>
                    <input
                      type="radio"
                      name="provider"
                      checked={provider === 'gemini'}
                      onChange={() => setProvider('gemini')}
                    />
                    <div>
                      <strong>🌟 Google Gemini (Cloud API)</strong>
                      <small>Fast, intelligent, runs via Gemini 3.8 Flash API key in backend/.env</small>
                    </div>
                  </label>
                </div>

                <div className="settings-option">
                  <label>
                    <input
                      type="radio"
                      name="provider"
                      checked={provider === 'ollama'}
                      onChange={() => setProvider('ollama')}
                    />
                    <div>
                      <strong>🦙 Ollama (Local Model)</strong>
                      <small>100% private, runs offline on your PC via http://localhost:11434</small>
                    </div>
                  </label>
                </div>

                <div className="settings-option">
                  <label>
                    <input
                      type="radio"
                      name="provider"
                      checked={provider === 'both'}
                      onChange={() => setProvider('both')}
                    />
                    <div>
                      <strong>⚡ Both (Side-by-Side Benchmark)</strong>
                      <small>Queries both Gemini and Ollama concurrently and displays answers side-by-side</small>
                    </div>
                  </label>
                </div>
              </div>

              <div className="modal-footer">
                <button className="primary-button" onClick={() => setShowSettings(false)}>Done</button>
              </div>
            </div>
          </div>
        )}

        <footer className="footer-note">
          Document Intelligence RAG · Choose between Gemini, Ollama, or Dual Execution
        </footer>
      </main>
    </div>
  )
}
