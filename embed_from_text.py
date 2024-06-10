text = '''
Q: MyITS saya bermasalah, bisa lapor kemana?

A: Silakan ajukan tiket ke DPTSI di https://servicedesk.its.ac.id/

Q: Saya ingin mengurus “…..” siapa tendik yang harus ditemui?

A: Temui tendik sesuai bidangnya yuk! :)

Persuratan (S1) → Pak Sugeng
Akademik (S1) → Bu Inka (Kelas, Nilai, MBKM, International Exposure), Bu Icha (Asisten Dosen, MonKP)
Akademik + Persuratan (S2) → Bu Kartika
Akademik + Persuratan (S3) → Bu Lina
Peminjaman Ruang → Pak Husein (Kasubbag)
Keuangan → Bu Devi, Bu Due
Urusan Ruang Aula → Pak Jumali
Urusan Ruang Kelas → Pak Edy
Urusan Laboratorium → Teknisi Lab. Pak Junaidi (PKT, KCV, LP), Pak Jumali (KBJ, RPL), Pak Gayuh (AP, MCI, GiGA), Pak Kunto (AJK).
Urusan Keamanan dan Kebersihan → Pak Sukron
Urusan Kelistrikan → Pak Hari
Q: Cara mendapatkan transkrip

A: Buka SIM Akademik, masuk ke menu LAPORAN -> TRANSKRIP

Q: Cara mendapatkan surat keterangan aktif mahasiswa

A: Buka SIM Akademik, masuk ke menu SURAT MAHASISWA -> LAYANAN SURAT MAHASISWA

Q: Bagaimana mendapatkan translasi ke dalam Bahasa Inggris dari dokumen resmi ITS?

A: Translasi dilayani oleh BURB
Keluhan/permintaan : servicedesk.its.ac.id
Email : plt@its.ac.id 
Telp : (031) 5994251
https://www.its.ac.id/burb/wp-content/uploads/sites/106/2022/05/Panduan-myITS-Services_rev-1.pdf"

Q: Untuk keperluan mendaftar bekerja / sekolah lanjutan, surat keterangan apakah yang bisa saya dapatkan?

A:
Sebelum yudisium
Surat keterangan telah menyelesaikan xxx sks dan telah selesai sidang TA
Setelah yudisium
Surat keterangan lulus yang diterbitkan oleh BURB
Setelah wisuda:
Ijazah dan transkrip resmi ITS

Q: Bagaimana cara mendapatkan Surat Keterangan Lulus, legalisir ijazah dll atau surat keterangan selsai TA?

A: SKL dilayani oleh BIRO UMUM DAN REFORMASI BIROKRASI (BURB)
https://www.its.ac.id/burb/id/permohonan-surat-keterangan-lulus/
SKL baru akan dikeluarkan jika sudah ada SK kelulusan dari Rektor ITS

Surat keterangan selesai TA diberikan oleh prodi T Informatika. Silakan kirim permohonan ke WA Pak Sugeng +62 856-4843-2445 disertai dengan data diri dan keperluan surat keterangan telah menyelesaikan TA
 

Q: Bagaimana cara berkomunikasi dengan email?

A: Email adalah media komunikasi resmi dari institusi, sedangkan WA bukan.
Tuliskan salam, isi email, dan ditutup dengan salam.
contoh adalah sebagai berikut
-------------------------------------------------------------------------------------------------------
Assalamualaikum/Selamat Pagi

Bersama dengan ini saya sertakan dokumen revisi TA saya dengan poin sebagai berikut:
- perbaikan ....
Mohon berkenan memeriksa dan menginformasikan jika ada yang perlu saya perbaiki.

Demikian, terima kasih
Agus
0511000..…

Q: Bagaimana mendapatkan surat keterangan yang tidak ada di menu integra, contoh surat keterangan selesai TA dan lulus >=144 SKS?

A: kirim ke WA Pak Sugeng +62 856-4843-2445
------------------------------------------------------------------------------
Kepada Yth
Bapak Ary Mazharuddin Shiddiqi
di tempat

Dengan hormat,
Dengan surat ini saya,

Nama: 
NRP: 
Tempat/Tgl lahir: 
Alamat sekarang: 
SKS Lulus:
email aktif:
No HP aktif:

Bermaksud untuk mengajukan permohonan kepada Bapak Ary guna memperoleh Surat Keterangan Selesai Tugas Akhir.

Demikian surat ini saya ajukan dan atas terkabulnya permohonan ini, saya ucapkan terimakasih

Surabaya, .…
'''

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

splitted_texts = text_splitter.split_text(text)

from langchain_community.vectorstores import FAISS
from app.model import load_model

embedding_model = load_model("./app/model/modules/indo-sentence-bert-base")

metadata = {"source": "Frequently Asks"}
metadatas = [metadata] * len(splitted_texts)
embedded_text = FAISS.from_texts(splitted_texts, embedding_model, metadatas=metadatas)
# embedded_text = FAISS.from_texts(splitted_texts, embedding_model)

print(embedded_text.docstore._dict)