from app.rag.chunker import DocumentChunker
from app.ingestion.models import ExtractedDocument,ExtractedPage

def doc(text):
    return ExtractedDocument(
        source_path="/tmp/a.txt",
        file_extension=".txt",
        pages=[ExtractedPage(page_number=1,text=text)],
        metadata={"title":"Sample"}
    )

def test_single_chunk():
    c=DocumentChunker(chunk_size=100,overlap=10)
    chunks=c.chunk(doc("hello world"))
    assert len(chunks)==1
    assert chunks[0].text=="hello world"

def test_overlap():
    c=DocumentChunker(chunk_size=10,overlap=2)
    chunks=c.chunk(doc("abcdefghijklmnopqrstuvwxyz"))
    assert len(chunks)>1
    assert chunks[1].start_char==8

def test_metadata():
    c=DocumentChunker()
    chunk=c.chunk(doc("abc"))[0]
    assert chunk.metadata["title"]=="Sample"
    assert chunk.page_number==1
