---
title: Thesis RAG
emoji: 🌿
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Ask a PhD thesis on mulberry genomics, with page citations
---

# Thesis RAG

Ask questions about a PhD thesis on SNP genotyping arrays in mulberry and get
answers that cite the thesis by chapter, section and page, with the reasoning
behind each answer. Questions the thesis does not cover are answered from
general knowledge and labelled as such.

Source code, evaluation and design notes: https://github.com/magic-peach/RAG-model

This demo runs Llama 3.1 8B through Groq; the full app also runs entirely
locally with Ollama. Uploading new documents is disabled here.
