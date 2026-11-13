let currentCorpus = [];
let currentCitations = [];

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    setupEventListeners();
    await loadCorpus();
    await checkInitialSeed();
}

function setupEventListeners() {
    // Ingest Modal
    const modal = document.getElementById("upload-modal");
    document.getElementById("btn-upload-modal").addEventListener("click", () => {
        modal.classList.add("open");
    });
    document.getElementById("btn-close-modal").addEventListener("click", () => {
        modal.classList.remove("open");
    });
    document.getElementById("btn-cancel-modal").addEventListener("click", () => {
        modal.classList.remove("open");
    });

    // Upload Form
    document.getElementById("upload-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        await handleDocumentUpload();
    });

    // Ask Query
    document.getElementById("btn-ask").addEventListener("click", () => {
        submitQuery();
    });

    document.getElementById("question-input").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            submitQuery();
        }
    });

    // Hint Pills
    const pills = document.querySelectorAll(".hint-pill");
    pills.forEach(p => {
        p.addEventListener("click", () => {
            document.getElementById("question-input").value = p.dataset.q;
            submitQuery();
        });
    });
}

async function loadCorpus() {
    try {
        const res = await fetch("/api/v1/corpus");
        if (!res.ok) throw new Error("Failed to load corpus");
        currentCorpus = await res.json();
        renderCorpusList();
        document.getElementById("corpus-stats").textContent = `${currentCorpus.length} Document${currentCorpus.length === 1 ? '' : 's'}`;
    } catch (err) {
        console.error("Corpus load error:", err);
    }
}

function renderCorpusList() {
    const list = document.getElementById("corpus-list");
    list.innerHTML = "";

    if (currentCorpus.length === 0) {
        list.innerHTML = `<div style="padding:16px; font-size:12px; color:var(--text-muted); text-align:center;">No documents uploaded. Click "+ Ingest Document" to populate knowledge base.</div>`;
        return;
    }

    currentCorpus.forEach(doc => {
        const card = document.createElement("div");
        card.className = "corpus-card";
        card.innerHTML = `
            <div class="corpus-card-title">${escapeHtml(doc.title)}</div>
            <div class="corpus-card-meta">
                <span>${doc.total_chunks} chunks • ${doc.word_count} words</span>
                <span style="color:var(--accent-teal);">${escapeHtml(doc.filename)}</span>
            </div>
        `;
        list.appendChild(card);
    });
}

async function handleDocumentUpload() {
    const title = document.getElementById("doc-title").value.trim();
    const filename = document.getElementById("doc-filename").value.trim() || "document.txt";
    const content = document.getElementById("doc-content").value.trim();

    if (!title || !content) return;

    try {
        const res = await fetch("/api/v1/corpus", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title, filename, content })
        });

        if (!res.ok) throw new Error("Document upload failed");

        document.getElementById("upload-modal").classList.remove("open");
        document.getElementById("upload-form").reset();
        await loadCorpus();
    } catch (err) {
        console.error("Upload error:", err);
        alert("Upload error: " + err.message);
    }
}

async function submitQuery() {
    const question = document.getElementById("question-input").value.trim();
    if (!question) return;

    const answerContainer = document.getElementById("answer-container");
    const warningBanner = document.getElementById("hallucination-warning");
    const metricsBar = document.getElementById("answer-metrics");

    answerContainer.innerHTML = `<div style="padding:24px; text-align:center; color:var(--accent-teal-light); font-family:var(--font-heading); font-size:13px;">Searching vector passages & evaluating citation grounding...</div>`;
    warningBanner.style.display = "none";
    metricsBar.style.display = "none";

    try {
        const res = await fetch("/api/v1/query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, top_k: 4 })
        });

        if (!res.ok) throw new Error("Query failed");
        const data = await res.json();

        metricsBar.style.display = "flex";
        document.getElementById("metric-confidence").textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        document.getElementById("metric-eval").textContent = `Evaluated: ${data.chunks_evaluated} Chunks`;

        currentCitations = data.citations || [];
        renderProvenanceList(currentCitations);

        if (data.hallucination_flag) {
            answerContainer.innerHTML = `<p style="color:var(--text-secondary); font-style:italic;">No verified response generated.</p>`;
            warningBanner.style.display = "flex";
            document.getElementById("hallucination-text").textContent = data.answer;
        } else {
            renderFormattedAnswer(data.answer);
        }

    } catch (err) {
        console.error("Query error:", err);
        answerContainer.innerHTML = `<div style="color:var(--accent-red); font-size:13px;">Query execution failed: ${escapeHtml(err.message)}</div>`;
    }
}

function renderFormattedAnswer(answerText) {
    const container = document.getElementById("answer-container");
    // Replace [1], [2] with interactive citation buttons
    const formatted = escapeHtml(answerText).replace(/\[(\d+)\]/g, (match, p1) => {
        return `<button class="citation-tag" data-cit="${p1}" title="View Citation [${p1}]">${p1}</button>`;
    });

    container.innerHTML = `<div class="answer-text">${formatted}</div>`;

    container.querySelectorAll(".citation-tag").forEach(btn => {
        btn.addEventListener("click", () => {
            const citNum = parseInt(btn.dataset.cit);
            highlightCitation(citNum);
        });
    });
}

function renderProvenanceList(citations) {
    const list = document.getElementById("citations-list");
    document.getElementById("citation-count").textContent = `${citations.length} Source${citations.length === 1 ? '' : 's'}`;
    list.innerHTML = "";

    if (citations.length === 0) {
        list.innerHTML = `<div class="empty-citations">No citations associated with the current query.</div>`;
        return;
    }

    citations.forEach(c => {
        const card = document.createElement("div");
        card.id = `cit-card-${c.citation_num}`;
        card.className = "citation-card";
        card.innerHTML = `
            <div class="citation-card-header">
                <span class="citation-source">[${c.citation_num}] ${escapeHtml(c.document_title)}</span>
                <span class="citation-score">Sim: ${(c.relevance_score * 100).toFixed(1)}%</span>
            </div>
            <div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>
            <div style="font-size:10px; font-family:var(--font-mono); color:var(--text-muted);">Chunk ID: ${escapeHtml(c.chunk_id)}</div>
        `;
        list.appendChild(card);
    });
}

function highlightCitation(citNum) {
    document.querySelectorAll(".citation-card").forEach(c => c.classList.remove("highlight"));
    const target = document.getElementById(`cit-card-${citNum}`);
    if (target) {
        target.classList.add("highlight");
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

async function checkInitialSeed() {
    if (currentCorpus.length === 0) {
        const seeds = [
            {
                title: "Distributed Consensus & Byzantine Fault Tolerance",
                filename: "consensus_principles.md",
                content: "In distributed systems, Byzantine Fault Tolerance (BFT) guarantees that a system can function correctly even when up to one-third of the nodes act maliciously or fail intermittently. The minimum threshold requires 3f + 1 total nodes to tolerate f faulty or Byzantine nodes. Practical Byzantine Fault Tolerance (PBFT) optimizes communication overhead through pre-prepare, prepare, and commit phases."
            },
            {
                title: "Quantum Error Correction & Surface Codes",
                filename: "quantum_surface_code.md",
                content: "Surface codes arrange physical qubits on a two-dimensional square lattice to protect logical qubits from bit-flip (X) and phase-flip (Z) errors. Syndrome extraction circuits measure stabilizer operators repetitively without destroying quantum coherence. The fault-tolerant error threshold for surface codes is approximately one percent under standard depolarizing noise models."
            }
        ];

        for (const s of seeds) {
            await fetch("/api/v1/corpus", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(s)
            });
        }
        await loadCorpus();
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
