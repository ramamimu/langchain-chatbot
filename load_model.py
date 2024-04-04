from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

model_kwargs = {'trust_remote_code': True}
embedding_firqaa = HuggingFaceEmbeddings(model_name='./models/indo-sentence-bert-base', model_kwargs=model_kwargs,)

UUD = PyPDFLoader('./docs/UUD-Nomor-Tahun-1945-UUD1945.pdf').load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

UUD_splits = text_splitter.split_documents(UUD)
UUD_fvs = FAISS.from_documents(UUD_splits, embedding_firqaa)

question = "Dimakah letak pasal yang berbunyi 'Majelis Permusyawaratan Rakyat terdiri atas anggota-anggota Dewan Perwakilan Rakyat, ditambah dengan utusan-utusan dari daerah-daerah dan golongan-golongan, menurut aturan yang ditetapkan dengan undang-undang.'"
ss = UUD_fvs.similarity_search(question, k=3)

print(ss)