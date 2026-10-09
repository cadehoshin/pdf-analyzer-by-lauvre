
import streamlit as st
import fitz
import ollama
import numpy as np

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="PDF Analyzer | by LAUVRE",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #0b1120;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.9);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #263247;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .brand {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #f8fafc;
        margin-bottom: 0;
    }

    .brand span {
        color: #818cf8;
    }

    .muted {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .hero {
        padding: 2.2rem 0 1.5rem 0;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.15;
        margin-bottom: 0.8rem;
    }

    .hero p {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.7;
    }

    .eyebrow {
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #172338, #111b2d);
        border: 1px solid #2a3850;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 100px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .section-label {
        color: #cbd5e1;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 1.4rem 0 0.7rem 0;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        border: 1px solid #334155;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: #6366f1;
        border-color: #6366f1;
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: #4f46e5;
        border-color: #4f46e5;
    }

    .stTextInput input,
    [data-testid="stChatInput"] textarea {
        background: #111b2d;
        color: #f8fafc;
        border-color: #334155;
        border-radius: 12px;
    }

    [data-testid="stChatMessage"] {
        background: #111b2d;
        border: 1px solid #263247;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    [data-testid="stExpander"] {
        background: #111b2d;
        border: 1px solid #263247;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        background: #111b2d;
        border: 1px dashed #475569;
        border-radius: 12px;
        padding: 0.7rem;
    }

    hr {
        border-color: #263247;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. SESSION STATE
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "filename" not in st.session_state:
    st.session_state["filename"] = None

if "file_key" not in st.session_state:
    st.session_state["file_key"] = None


# ==========================================
# 3. FUNGSI PEMROSESAN DOKUMEN
# ==========================================


def create_chunks(pages, chunk_size=180, overlap=40):
    chunks = []

    # Pastikan setiap langkah maju tetap positif
    step = chunk_size - overlap

    if step <= 0:
        raise ValueError(
            "Overlap harus lebih kecil dari chunk_size."
        )

    for page_number, page_text in enumerate(pages, start=1):
        words = page_text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                })

            # Tidak perlu membuat potongan tambahan
            # jika bagian teks berikutnya sudah kosong.
            if start + chunk_size >= len(words):
                break

    return chunks


    for page_number, page_text in enumerate(pages, start=1):
        words = page_text.split()

        for start in range(0, len(words), chunk_size):
            chunk_text = " ".join(
                words[start:start + chunk_size]
            )

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                })

    return chunks


def get_embedding(text):
    result = ollama.embed(
        model="nomic-embed-text",
        input=text,
    )

    return np.array(result["embeddings"][0])


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ==========================================
# 4. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown(
        '<p class="brand">PDF<span>Analyzer</span></p>',
        unsafe_allow_html=True,
    )

    st.caption("Document Analyzer | by LAUVRE")

    st.divider()

    st.markdown("#### Workspace")
    st.markdown("📄 **Document Q&A**")
    st.caption("Tanyakan isi dokumen menggunakan AI lokal.")

    st.markdown(
        '<p class="section-label">DOKUMEN</p>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Unggah dokumen PDF",
        type=["pdf"],
        help="Pilih dokumen PDF yang ingin dianalisis.",
    )

    current_file_key = None

    if uploaded_file is not None:
        current_file_key = (
            f"{uploaded_file.name}:{uploaded_file.size}"
        )

        st.caption(f"File: {uploaded_file.name}")

        if (
            st.session_state["file_key"] != current_file_key
        ):
            st.info(
                "Dokumen baru dipilih. Proses dokumen "
                "untuk mulai bertanya."
            )

            if st.button(
                "Proses dokumen",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Membaca dan memproses PDF..."
                    ):
                        document = fitz.open(
                            stream=uploaded_file.getvalue(),
                            filetype="pdf",
                        )

                        pages = [
                            page.get_text()
                            for page in document
                        ]

                        if not any(
                            page.strip() for page in pages
                        ):
                            st.error(
                                "Teks tidak ditemukan. "
                                "PDF mungkin berupa hasil scan."
                            )
                        else:
                            new_chunks = create_chunks(pages)

                            for chunk in new_chunks:
                                chunk["embedding"] = (
                                    get_embedding(
                                        chunk["text"]
                                    ).tolist()
                                )

                            st.session_state["chunks"] = new_chunks
                            st.session_state["filename"] = (
                                uploaded_file.name
                            )
                            st.session_state["file_key"] = (
                                current_file_key
                            )
                            st.session_state["chat_history"] = []

                            st.rerun()

                except Exception as error:
                    st.error(
                        f"Gagal memproses dokumen: {error}"
                    )

        else:
            st.success("Dokumen siap digunakan.")

            if st.button(
                "Proses ulang dokumen",
                use_container_width=True,
            ):
                st.session_state["file_key"] = None
                st.session_state["chunks"] = []
                st.session_state["chat_history"] = []
                st.rerun()

    else:
        st.caption("Belum ada dokumen yang dipilih.")

    st.divider()

    st.markdown("#### Model AI")

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">LANGUAGE MODEL</div>
            <div class="metric-value">Llama 3.2</div>
            <div class="muted">Powered by Ollama</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">EMBEDDING MODEL</div>
            <div class="metric-value">Nomic Embed</div>
            <div class="muted">Semantic search</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    if st.button(
        "＋  Percakapan baru",
        use_container_width=True,
    ):
        st.session_state["chat_history"] = []
        st.rerun()

    st.markdown(
        '<p class="muted">PDF Analyzer · Local AI Project</p>',
        unsafe_allow_html=True,
    )


# ==========================================
# 5. HEADER DAN STATUS
# ==========================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">YOUR PERSONAL DOCUMENT ASSISTANT</div>
        <h1>Chat with your documents.</h1>
        <p>
            Temukan informasi penting dari PDF, pahami isi dokumen,
            dan ajukan pertanyaan menggunakan AI yang berjalan lokal.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">STATUS</div>
            <div class="metric-value">● Local AI</div>
            <div class="muted">Pemrosesan melalui Ollama</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    doc_status = (
        "Siap bertanya"
        if st.session_state["file_key"] is not None
        else "Menunggu PDF"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">DOKUMEN</div>
            <div class="metric-value">{doc_status}</div>
            <div class="muted">PDF knowledge base</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    total_chunks = len(st.session_state["chunks"])

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">TEXT CHUNKS</div>
            <div class="metric-value">{total_chunks}</div>
            <div class="muted">Potongan teks terindeks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ==========================================
# 6. AREA PERCAKAPAN
# ==========================================

if st.session_state["file_key"] is None:
    st.markdown("### Mulai dengan sebuah dokumen")

    st.write(
        "Unggah PDF melalui sidebar di sebelah kiri, "
        "lalu klik **Proses dokumen**."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">01 · Upload</div>
                <p class="muted">
                    Pilih dokumen PDF dari komputer kamu.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">02 · Ask</div>
                <p class="muted">
                    Ajukan pertanyaan dan temukan jawabannya.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.markdown("### Percakapan")

    st.caption(
        f"Dokumen aktif: {st.session_state['filename']}"
    )

    # Input chat
    question = st.chat_input(
        "Tanyakan sesuatu tentang dokumen ini..."
    )

    # Proses pertanyaan baru
    if question:
        try:
            with st.spinner(
                "Mencari informasi yang relevan..."
            ):
                question_embedding = get_embedding(question)

                scored_chunks = []

                for chunk in st.session_state["chunks"]:
                    score = cosine_similarity(
                        question_embedding,
                        np.array(chunk["embedding"]),
                    )

                    scored_chunks.append((score, chunk))

                scored_chunks.sort(
                    key=lambda item: item[0],
                    reverse=True,
                )

                relevant_chunks = [
                    chunk
                    for score, chunk in scored_chunks[:3]
                ]

                context = "\n\n".join(
                    f"Halaman {chunk['page']}:\n{chunk['text']}"
                    for chunk in relevant_chunks
                )

            with st.spinner("Menyusun jawaban..."):
                response = ollama.chat(
                    model="llama3.2",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Kamu adalah asisten dokumen. "
                                "Jawab menggunakan informasi dalam "
                                "konteks yang diberikan. "
                                "Gunakan bahasa Indonesia yang jelas. "
                                "Jika jawabannya tidak ditemukan dalam "
                                "konteks, katakan dengan jujur. "
                                "Sertakan sumber halaman jika relevan, "
                                "dengan format (Sumber: Halaman X). "
                                "Jangan mengarang isi atau nomor halaman."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Konteks dokumen:\n{context}\n\n"
                                f"Pertanyaan: {question}"
                            ),
                        },
                    ],
                )

            answer = response["message"]["content"]

            # Simpan sumber yang digunakan
            sources = []

            for chunk in relevant_chunks:
                source = {
                    "page": chunk["page"],
                    "text": chunk["text"],
                }

                if source not in sources:
                    sources.append(source)

            # Simpan riwayat percakapan
            st.session_state["chat_history"].append({
                "question": question,
                "answer": answer,
                "sources": sources,
            })

        except Exception as error:
            st.error(f"Terjadi kesalahan: {error}")

    # Tampilkan riwayat
    if not st.session_state["chat_history"]:
        st.info(
            "Dokumen sudah siap. Mulai dengan mengetik "
            "pertanyaan pada kolom chat di bawah."
        )

        st.markdown("**Contoh pertanyaan**")

        suggestions = [
            "Apa inti dokumen ini?",
            "Apa kesimpulan utamanya?",
            "Sebutkan poin-poin penting dokumen.",
        ]

        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    for chat in st.session_state["chat_history"]:
        with st.chat_message("user", avatar="👤"):
            st.write(chat["question"])

        with st.chat_message("assistant", avatar="📚"):
            st.markdown(chat["answer"])

            with st.expander("📎 Lihat sumber jawaban"):
                for source in chat.get("sources", []):
                    st.markdown(
                        f"**Halaman {source['page']}**"
                    )
                    st.write(source["text"])
                    st.divider()

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b;
                font-size:0.8rem; padding:0.5rem 0;">
        PDF Analyzer · Built by LAUVRE
    </div>
    """,
    unsafe_allow_html=True,
)
