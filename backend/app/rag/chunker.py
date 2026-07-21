from __future__ import annotations

from dataclasses import dataclass
from app.ingestion.models import ExtractedDocument

@dataclass(slots=True)
class TextChunk:
    chunk_index:int
    page_number:int
    text:str
    start_char:int
    end_char:int
    metadata:dict

class DocumentChunker:
    def __init__(self, chunk_size:int=800, overlap:int=120):
        if overlap>=chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size=chunk_size
        self.overlap=overlap

    def chunk(self, document:ExtractedDocument)->list[TextChunk]:
        chunks=[]
        idx=0
        for page in document.pages:
            text=page.text
            start=0
            while start < len(text):
                end=min(start+self.chunk_size,len(text))
                chunks.append(TextChunk(
                    chunk_index=idx,
                    page_number=page.page_number,
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                    metadata={
                        "source_path":document.source_path,
                        "file_extension":document.file_extension,
                        **document.metadata,
                    }
                ))
                idx+=1
                if end==len(text):
                    break
                start=end-self.overlap
        return chunks
