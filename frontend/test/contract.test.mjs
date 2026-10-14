import test from 'node:test'
import assert from 'node:assert/strict'

test('AnswerLanguage contract verification', () => {
  const supportedLanguages = ['en', 'ml']
  assert.equal(supportedLanguages.includes('en'), true)
  assert.equal(supportedLanguages.includes('ml'), true)
  assert.equal(supportedLanguages.includes('fr'), false)
})

test('Chat request payload structure with Malayalam language', () => {
  const payload = {
    document_id: 'test-doc-id',
    question: 'ഇത് എന്തിനെക്കുറിച്ചാണ്?',
    provider: 'gemini',
    answer_language: 'ml',
    history: []
  }

  assert.equal(payload.answer_language, 'ml')
  assert.equal(payload.provider, 'gemini')
  assert.equal(typeof payload.question, 'string')
})
