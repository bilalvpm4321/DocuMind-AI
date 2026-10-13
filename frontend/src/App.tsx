import { useRef, useState, useEffect } from 'react'
import {
  FileText, Upload, Send, LoaderCircle, Bot, UserRound,
  FileUp, Sparkles, Trash2, Cpu, Cloud, Layers, Settings, X,
  Globe, Languages, ChevronDown
} from 'lucide-react'

type Source = { page: number | null; kind: string; text: string; score: number }
export type AnswerLanguage = 'en' | 'ml'

type Message = {
  role: 'user' | 'assistant'
  content: string
  provider?: 'gemini' | 'ollama' | 'both'
  answer_language?: AnswerLanguage
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
  const [answerLanguage, setAnswerLanguage] = useState<AnswerLanguage>('en')
  const [showSettings, setShowSettings] = useState(false)
  const [showScrollBottom, setShowScrollBottom] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const chatStreamRef = useRef<HTMLElement>(null)

  const scrollToBottom = (smooth = true) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' })
  }

  useEffect(() => {
    scrollToBottom(true)
  }, [messages, sending])

  const handleScroll = () => {
    if (!chatStreamRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = chatStreamRef.current
    const isUp = scrollHeight - scrollTop - clientHeight > 100
    setShowScrollBottom(isUp)
  }

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
        content: `I've processed **${data.filename}** and indexed ${data.chunks} text chunks. You can now ask questions using Gemini, Ollama, or run both to compare answers side-by-side in English or Malayalam!`
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
          answer_language: answerLanguage,
          language: answerLanguage,
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
          answer_language: data.answer_language || answerLanguage,
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
          content: answerLanguage === 'ml'
            ? 'ഈ ചോദ്യത്തിന് ഉത്തരം കണ്ടെത്താൻ കഴിഞ്ഞില്ല. സെർവർ ക്രമീകരണങ്ങളും കണക്ഷനും പരിശോധിക്കുക.'
            : 'I could not answer that question. Please verify your selected provider settings and backend connection.',
          provider: provider,
          answer_language: answerLanguage,
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
          <div className="brand-mark"><Bot size={22} /></div>
          <div>
            <strong>DocuMind AI</strong>
            <span>Document Intelligence RAG</span>
          </div>
        </div>

        <button className="new-doc" onClick={reset} title="Start new document session">
          <Trash2 size={15} /> New Session
        </button>

        <div className="side-label">ENGINE SELECTION</div>
        <div className="provider-selector-sidebar">
          <button
            type="button"
            className={`provider-tab ${provider === 'gemini' ? 'active' : ''}`}
            onClick={() => setProvider('gemini')}
          >
            <Cloud size={16} />
            <div>
              <strong>Google Gemini</strong>
              <small>Cloud API · Fast</small>
            </div>
          </button>

          <button
            type="button"
            className={`provider-tab ${provider === 'ollama' ? 'active' : ''}`}
            onClick={() => setProvider('ollama')}
          >
            <Cpu size={16} />
            <div>
              <strong>Local Ollama</strong>
              <small>Offline · 100% Private</small>
            </div>
          </button>

          <button
            type="button"
            className={`provider-tab ${provider === 'both' ? 'active' : ''}`}
            onClick={() => setProvider('both')}
          >
            <Layers size={16} />
            <div>
              <strong>Dual Comparison</strong>
              <small>Side-by-Side Execution</small>
            </div>
          </button>
        </div>

        <div className="side-label">RESPONSE LANGUAGE</div>
        <div className="lang-selector-sidebar" role="radiogroup" aria-label="Sidebar language selector">
          <button
            type="button"
            role="radio"
            aria-checked={answerLanguage === 'en'}
            className={`sidebar-lang-tab ${answerLanguage === 'en' ? 'active' : ''}`}
            onClick={() => setAnswerLanguage('en')}
          >
            <Globe size={15} />
            <div>
              <strong>English</strong>
              <small>Standard grounded answers</small>
            </div>
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={answerLanguage === 'ml'}
            className={`sidebar-lang-tab ml-tab ${answerLanguage === 'ml' ? 'active active-ml' : ''}`}
            onClick={() => setAnswerLanguage('ml')}
          >
            <Languages size={15} />
            <div>
              <strong>മലയാളം (Malayalam)</strong>
              <small>മലയാളം ലിപി · Localized Q&A</small>
            </div>
          </button>
        </div>

        <div className="side-label">ACTIVE DOCUMENT</div>
        {document ? (
          <div className="active-doc-card">
            <div className="doc-icon"><FileText size={18} /></div>
            <div className="doc-meta">
              <span className="doc-name" title={document.filename}>{document.filename}</span>
              <span className="doc-sub">{document.pages_with_text} pages · {document.chunks} chunks</span>
            </div>
          </div>
        ) : (
          <p className="empty-sidebar">No document loaded. Upload a PDF, Word doc, or image to start chatting.</p>
        )}

        <div className="sidebar-bottom">
          <button className="settings-trigger" onClick={() => setShowSettings(true)}>
            <Settings size={14} /> Settings
          </button>
          <span className="version">v0.2.0 ML</span>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="main-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">RAG DOCUMENT CHAT</span>
            <h1>Chat with your document</h1>
          </div>

          <div className="topbar-controls">
            {/* Language Selector Segmented Control */}
            <div className="lang-toggle-group" role="radiogroup" aria-label="Select answer language">
              <button
                type="button"
                role="radio"
                aria-checked={answerLanguage === 'en'}
                className={`lang-pill-btn ${answerLanguage === 'en' ? 'active' : ''}`}
                onClick={() => setAnswerLanguage('en')}
                title="Generate answers in English"
              >
                <Globe size={13} /> English
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={answerLanguage === 'ml'}
                className={`lang-pill-btn ${answerLanguage === 'ml' ? 'active active-ml' : ''}`}
                onClick={() => setAnswerLanguage('ml')}
                title="മലയാളത്തിൽ ഉത്തരം നൽകുക"
              >
                <Languages size={13} /> മലയാളം
              </button>
            </div>

            {/* Quick Provider Toggle */}
            <div className="pill-group" role="radiogroup" aria-label="Select model provider">
              <button
                type="button"
                role="radio"
                aria-checked={provider === 'gemini'}
                className={`pill-btn ${provider === 'gemini' ? 'active' : ''}`}
                onClick={() => setProvider('gemini')}
                title="Use Google Gemini API"
              >
                <Cloud size={13} /> Gemini
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={provider === 'ollama'}
                className={`pill-btn ${provider === 'ollama' ? 'active' : ''}`}
                onClick={() => setProvider('ollama')}
                title="Use Local Ollama"
              >
                <Cpu size={13} /> Ollama
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={provider === 'both'}
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
              Switch seamlessly between <strong>English</strong> and <strong>മലയാളം</strong>, and choose between <strong>Google Gemini</strong>, <strong>Local Ollama</strong>, or <strong>Both</strong>.
            </p>

            <div
              className={`dropzone ${uploading ? 'uploading' : ''}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault()
                void uploadFile(e.dataTransfer.files[0])
              }}
            >
              <input
                ref={fileInput}
                type="file"
                hidden
                accept=".pdf,.docx,.txt,.png,.jpg,.jpeg,.webp"
                onChange={e => void uploadFile(e.target.files?.[0])}
              />
              <div className="dropzone-inner">
                {uploading ? (
                  <>
                    <LoaderCircle size={32} className="spin dropzone-spin" />
                    <strong>Extracting & Indexing document…</strong>
                    <span>Running PyMuPDF, OCR fallback, and FAISS vector embeddings</span>
                  </>
                ) : (
                  <>
                    <Upload size={32} />
                    <strong>Drop a document here, or browse</strong>
                    <span>Supports PDF, DOCX, TXT, PNG, JPG, JPEG, WEBP (up to 20MB)</span>
                  </>
                )}
              </div>
            </div>
          </section>
        ) : (
          <>
            <section
              className="chat-stream chat-area"
              ref={chatStreamRef}
              onScroll={handleScroll}
              aria-live="polite"
            >
              {messages.map((message, i) => (
                <div key={i} className={`message message-row ${message.role}`}>
                  <div className="avatar">
                    {message.role === 'assistant' ? <Bot size={18} /> : <UserRound size={18} />}
                  </div>

                  <div className="message-body">
                    <div className="message-header-row">
                      <span className="message-label">
                        {message.role === 'assistant' ? 'DOCUMENT AI' : 'YOU'}
                      </span>
                      {message.role === 'assistant' && (
                        <div className="badges-group">
                          {message.provider && (
                            <span className={`provider-badge ${message.provider}`}>
                              {message.provider === 'gemini' && '🌟 Gemini Flash'}
                              {message.provider === 'ollama' && '🦙 Ollama Local'}
                              {message.provider === 'both' && '⚡ Gemini & Ollama Comparison'}
                            </span>
                          )}
                          {message.answer_language && (
                            <span className={`lang-badge ${message.answer_language}`}>
                              {message.answer_language === 'ml' ? '🌐 മലയാളം' : '🌐 English'}
                            </span>
                          )}
                        </div>
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
                </div>
              ))}

              {sending && (
                <div className="thinking">
                  <LoaderCircle size={16} className="spin" />
                  {answerLanguage === 'ml'
                    ? (provider === 'both'
                        ? 'രേഖ പരിശോധിച്ച് Gemini-യും Ollama-യും ഉപയോഗിച്ച് മലയാളത്തിൽ മറുപടി തയ്യാറാക്കുന്നു…'
                        : `രേഖ പരിശോധിച്ച് ${provider === 'gemini' ? 'Gemini' : 'Ollama'} വഴി മലയാളത്തിൽ മറുപടി തയ്യാറാക്കുന്നു…`)
                    : (provider === 'both'
                        ? 'Querying both Gemini and Ollama in parallel…'
                        : `Searching document and asking ${provider === 'gemini' ? 'Gemini' : 'Ollama'}…`)}
                </div>
              )}
              <div ref={messagesEndRef} />
            </section>

            {showScrollBottom && (
              <button
                type="button"
                className="scroll-bottom-btn"
                onClick={() => scrollToBottom(true)}
                title="Scroll down to latest response"
                aria-label="Scroll to bottom"
              >
                <ChevronDown size={15} />
                <span>Jump to latest</span>
              </button>
            )}

            <div className="chat-composer">
              <form onSubmit={e => { e.preventDefault(); void sendQuestion() }}>
                <textarea
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  placeholder={answerLanguage === 'ml'
                    ? `രേഖയെക്കുറിച്ച് ചോദ്യങ്ങൾ ചോദിക്കുക (${provider === 'both' ? 'Gemini & Ollama' : provider === 'gemini' ? 'Gemini' : 'Ollama'})…`
                    : `Ask a question with ${provider === 'both' ? 'Gemini & Ollama' : provider === 'gemini' ? 'Gemini' : 'Ollama'}…`}
                  rows={2}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      void sendQuestion()
                    }
                  }}
                />
                <div className="composer-footer">
                  <div className="composer-footer-left">
                    <div className="composer-provider-hint">
                      Engine: <strong>{provider.toUpperCase()}</strong>
                    </div>
                    <button
                      type="button"
                      className={`composer-lang-pill ${answerLanguage === 'ml' ? 'ml-active' : ''}`}
                      onClick={() => setAnswerLanguage(prev => prev === 'en' ? 'ml' : 'en')}
                      title="Click to toggle response language between English and Malayalam"
                    >
                      <Languages size={13} />
                      <span>Answer: <strong>{answerLanguage === 'en' ? 'English' : 'മലയാളം'}</strong></span>
                    </button>
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
                <h3><Settings size={18} /> Model & Language Configuration</h3>
                <button className="icon-button" onClick={() => setShowSettings(false)}><X size={18} /></button>
              </div>

              <div className="modal-body">
                <h4>Answer Language (മറുപടി ഭാഷ)</h4>
                <div className="modal-lang-grid" role="radiogroup" aria-label="Modal language selection">
                  <label className={`lang-modal-card ${answerLanguage === 'en' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="modalLanguage"
                      checked={answerLanguage === 'en'}
                      onChange={() => setAnswerLanguage('en')}
                    />
                    <div>
                      <strong>🇺🇸 English (Default)</strong>
                      <small>Answers synthesized in clear English with page citations.</small>
                    </div>
                  </label>

                  <label className={`lang-modal-card ml-card ${answerLanguage === 'ml' ? 'selected' : ''}`}>
                    <input
                      type="radio"
                      name="modalLanguage"
                      checked={answerLanguage === 'ml'}
                      onChange={() => setAnswerLanguage('ml')}
                    />
                    <div>
                      <strong>🌴 മലയാളം (Malayalam)</strong>
                      <small>Answers written in natural Malayalam script (മലയാളം ലിപി) with accurate facts.</small>
                    </div>
                  </label>
                </div>

                <h4 style={{ marginTop: 12 }}>LLM Engine Architecture</h4>
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
          DocuMind AI · Retrieval-Augmented Generation · Multi-LLM & Malayalam Mode
        </footer>
      </main>
    </div>
  )
}
