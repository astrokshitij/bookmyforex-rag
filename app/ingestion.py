import os
import re
import yaml
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        text: str,
        source_file: str,
        document_title: str,
        document_type: str,
        section_title: str,
        last_updated: str,
        metadata: Dict[str, Any]
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source_file = source_file
        self.document_title = document_title
        self.document_type = document_type
        self.section_title = section_title
        self.last_updated = last_updated
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "document_title": self.document_title,
            "document_type": self.document_type,
            "section_title": self.section_title,
            "last_updated": self.last_updated,
            "metadata": self.metadata
        }

def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extracts YAML frontmatter delimited by --- at start of markdown file.
    Returns (metadata_dict, remaining_body_markdown).
    """
    frontmatter_pattern = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    match = frontmatter_pattern.match(content)
    if match:
        yaml_text = match.group(1)
        try:
            metadata = yaml.safe_load(yaml_text) or {}
        except Exception:
            metadata = {}
        body = content[match.end():]
        return metadata, body
    return {}, content

def clean_citations_and_markup(text: str) -> str:
    """Cleans citation tags like [cite: 1] while preserving content and structure."""
    # Keep citations clean or normalize them
    cleaned = re.sub(r'\[cite:\s*\d+\]', '', text)
    # Normalize multiple blank lines to at most two
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()

def chunk_markdown_preserving_clauses(
    file_path: Path,
    metadata: Dict[str, Any],
    body: str
) -> List[DocumentChunk]:
    """
    Chunks markdown text into semantic sections and legal clauses,
    ensuring lists, slabs, and policy rules stay intact.
    """
    source_file = file_path.name
    doc_title = metadata.get("document_title") or file_path.stem.replace("-", " ").title()
    doc_type = metadata.get("document_type") or "general_kb"
    last_updated = str(metadata.get("last_updated", "unknown"))

    # Split body into sections by markdown headings (# or ## or ###) or horizontal rules (---)
    # We want to identify headings while keeping the heading text
    lines = body.split("\n")
    
    sections: List[Dict[str, Any]] = []
    current_h1 = ""
    current_h2 = ""
    current_h3 = ""
    current_lines: List[str] = []
    current_title = "Overview"

    for line in lines:
        stripped = line.strip()
        
        # Check if line is a divider
        if stripped in ("---", "***", "___") and len(current_lines) > 0:
            section_content = "\n".join(current_lines).strip()
            if section_content:
                sections.append({
                    "title": current_title,
                    "h1": current_h1,
                    "h2": current_h2,
                    "h3": current_h3,
                    "content": section_content
                })
            current_lines = []
            continue

        # Check for headers
        h1_match = re.match(r"^#\s+(.+)$", stripped)
        h2_match = re.match(r"^##\s+(.+)$", stripped)
        h3_match = re.match(r"^###\s+(.+)$", stripped)

        if h1_match:
            # Flush existing
            if current_lines:
                section_content = "\n".join(current_lines).strip()
                if section_content:
                    sections.append({
                        "title": current_title,
                        "h1": current_h1,
                        "h2": current_h2,
                        "h3": current_h3,
                        "content": section_content
                    })
                current_lines = []
            current_h1 = h1_match.group(1).strip()
            current_h2 = ""
            current_h3 = ""
            current_title = current_h1
            current_lines.append(line)
        elif h2_match:
            # Flush existing if we already have content under prior h2/h1
            if current_lines:
                section_content = "\n".join(current_lines).strip()
                if section_content:
                    sections.append({
                        "title": current_title,
                        "h1": current_h1,
                        "h2": current_h2,
                        "h3": current_h3,
                        "content": section_content
                    })
                current_lines = []
            current_h2 = h2_match.group(1).strip()
            current_h3 = ""
            current_title = f"{current_h1} > {current_h2}" if current_h1 else current_h2
            current_lines.append(line)
        elif h3_match:
            # Check if current block has significant content before switching
            if len("\n".join(current_lines)) > 400:
                section_content = "\n".join(current_lines).strip()
                if section_content:
                    sections.append({
                        "title": current_title,
                        "h1": current_h1,
                        "h2": current_h2,
                        "h3": current_h3,
                        "content": section_content
                    })
                current_lines = []
            current_h3 = h3_match.group(1).strip()
            breadcrumbs = [b for b in [current_h1, current_h2, current_h3] if b]
            current_title = " > ".join(breadcrumbs)
            current_lines.append(line)
        else:
            current_lines.append(line)

    # Flush remaining lines
    if current_lines:
        section_content = "\n".join(current_lines).strip()
        if section_content:
            sections.append({
                "title": current_title,
                "h1": current_h1,
                "h2": current_h2,
                "h3": current_h3,
                "content": section_content
            })

    chunks: List[DocumentChunk] = []
    chunk_index = 0

    for sec in sections:
        raw_text = sec["content"]
        cleaned_text = clean_citations_and_markup(raw_text)
        if not cleaned_text or len(cleaned_text) < 20:
            continue

        sec_title = sec["title"] or "General"
        
        # Build contextual representation:
        # Prepending Document and Section metadata gives the embedding model and LLM rich grounding
        header_context = f"Document: {doc_title} ({source_file})\nSection: {sec_title}\nDocument Type: {doc_type}\nLast Updated: {last_updated}\n\n"
        full_chunk_text = header_context + cleaned_text

        # Create unique, deterministic ID
        hasher = hashlib.md5()
        hasher.update(f"{source_file}_{chunk_index}_{sec_title}".encode('utf-8'))
        chunk_id = f"{file_path.stem}_{chunk_index}_{hasher.hexdigest()[:8]}"

        meta_dict = {
            "source_file": source_file,
            "document_title": doc_title,
            "document_type": doc_type,
            "section_title": sec_title,
            "last_updated": last_updated,
            "status": str(metadata.get("status", "active")),
            "chunk_index": chunk_index
        }

        chunk = DocumentChunk(
            chunk_id=chunk_id,
            text=full_chunk_text,
            source_file=source_file,
            document_title=doc_title,
            document_type=doc_type,
            section_title=sec_title,
            last_updated=last_updated,
            metadata=meta_dict
        )
        chunks.append(chunk)
        chunk_index += 1

    return chunks

def load_and_chunk_all_markdown(workspace_dir: Optional[Path] = None) -> List[DocumentChunk]:
    """
    Scans the workspace directory for all *.md files,
    parses YAML frontmatter, and extracts clause-preserved chunks.
    """
    target_dir = workspace_dir or settings.BASE_DIR
    md_files = list(target_dir.glob("*.md"))
    
    # Filter out README.md or agent-generated documentation if any
    filtered_files = [f for f in md_files if f.name.lower() not in ("readme.md", "walkthrough.md", "implementation_plan.md")]

    all_chunks: List[DocumentChunk] = []
    for md_file in filtered_files:
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            metadata, body = parse_frontmatter(content)
            chunks = chunk_markdown_preserving_clauses(md_file, metadata, body)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error reading and chunking {md_file}: {e}")

    return all_chunks

if __name__ == "__main__":
    print("Testing Markdown Ingestion & Chunking...")
    chunks = load_and_chunk_all_markdown()
    print(f"Total chunks extracted: {len(chunks)}")
    for i, c in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1}: [{c.source_file}] {c.section_title} ({len(c.text)} chars) ---")
