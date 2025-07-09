import pandas as pd
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def baris():
    print("-----------------------------------------------")

baris() 
print("                Perkenalan DiriKu")
baris()

def text_to_speech(text):
   # Inisialisasi pyttsx3
    engine = pyttsx3.init()

    # Mengubah teks menjadi suara dan langsung memutarnya
    engine.say(text)
    engine.runAndWait()
    
def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Silakan berbicara...")
        
        # Menyesuaikan dengan kebisingan latar belakang
        recognizer.adjust_for_ambient_noise(source)

        # Mendengarkan suara dari mikrofon
        audio = recognizer.listen(source)

    try:
        # Mengonversi suara menjadi teks menggunakan Google Speech Recognition
        text = recognizer.recognize_google(audio, language='id-ID')  # Menggunakan bahasa Indonesia
        print(f"Anda berkata: {text}")
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition tidak dapat mengenali ucapan.")
        return None
    except sr.RequestError:
        print("Tidak dapat menghubungi layanan Google Speech Recognition.")
        return None

# Data Q&A
qa_data = [
    # Nama dan Panggilan
    {"question": "Siapa nama lengkap kamu?", "answer": "Nama lengkap saya Popopow, biasa dipanggil Popo."},
    {"question": "Nama panggilan kamu apa?", "answer": "Nama panggilan saya Popo, tapi kadang juga dipanggil Popopow."},
    {"question": "Biasa dipanggil siapa?", "answer": "Saya biasa dipanggil Popo atau Popopow."},

    # Usia
    {"question": "Berapa usia kamu?", "answer": "Saya berusia 19 tahun."},
    {"question": "Kamu lahir tahun berapa?", "answer": "Saya lahir pada tahun 2006, jadi sekarang saya berusia 19 tahun."},

    # Pendidikan dan Sekolah
    {"question": "Kamu kuliah atau sudah bekerja?", "answer": "Saya masih kuliah, baru saja diterima di jurusan Teknologi Informasi lewat snbt."},
    {"question": "Kamu kuliah di mana", "answer": "Saya kuliah di Institut Teknologi Sepuluh Nopember"},
    {"question": "Apa yang membuat kamu memilih untuk kuliah di jurusan Teknologi Informasi?", "answer": "Karena jurusan ini mencakup berbagai bidang seperti AI, Cybersecurity, Web, IoT, dan banyak lagi, yang membuka banyak peluang karir di bidang teknologi."},
    {"question": "Kenapa kamu memilih Institut Teknologi Sepuluh Nopember untuk kuliah?", "answer": "Saya memilih ITS karena reputasinya yang sangat baik di bidang teknologi, dan kampusnya juga sangat mendukung perkembangan ilmu pengetahuan dan teknologi."},

    # Jurusan dan Bidang Studi
    {"question": "Kamu kuliah di jurusan apa?", "answer": "Saya kuliah di jurusan Teknologi Informasi."},
    {"question": "Kenapa memilih jurusan Teknologi Informasi?", "answer": "Karena jurusan ini menawarkan kurikulum yang mencakup berbagai bidang seperti AI, Cybersecurity, Web Development, IoT, dan Embedded Systems, yang memungkinkan saya untuk mengeksplorasi banyak hal di dunia teknologi."},
    {"question": "Apa yang paling menarik dari kurikulum jurusan Teknologi Informasi di ITS?", "answer": "Yang paling menarik adalah adanya mata kuliah yang sangat relevan dengan perkembangan teknologi, seperti AI, Cloud Computing, Cybersecurity, dan Embedded Systems."},
    {"question": "Apa pelajaran yang paling kamu minati di jurusan ini?", "answer": "Pelajaran yang paling saya minati adalah Artificial Intelligence (AI) dan Cybersecurity."},
    {"question": "Apa bidang yang paling kamu minati antara AI, Cybersecurity, dan Robotic?", "answer": "Saat ini, saya masih bimbang antara AI dan Cybersecurity, tapi saya juga tertarik pada dunia robotik."},

    # Hobi
    {"question": "Hobi kamu apa?", "answer": "Hobi saya adalah bermain game."},
    {"question": "Selain kuliah, apa kegiatan yang paling sering kamu lakukan?", "answer": "Selain kuliah, saya sering bermain game, membaca buku, dan menonton film."},
    {"question": "Kenapa kamu suka bermain game?", "answer": "Saya suka bermain game karena dapat mengasah kemampuan berpikir dan strategi, serta memberikan hiburan."},
    {"question": "Game apa yang paling sering kamu mainkan?", "answer": "Game yang sering saya mainkan adalah League of Legends dan berbagai game RPG lainnya."},

    # Minat dalam Teknologi
    {"question": "Apa yang membuat kamu tertarik dengan dunia teknologi?", "answer": "Saya tertarik pada teknologi karena dapat mempermudah kehidupan sehari-hari dan memungkinkan kita untuk menciptakan hal-hal baru yang bermanfaat."},
    {"question": "Kenapa teknologi seperti AI dan Cybersecurity menarik bagi kamu?", "answer": "AI sangat menarik karena dapat mengubah berbagai aspek kehidupan dengan otomatisasi dan analisis data yang cerdas, sedangkan Cybersecurity penting untuk melindungi data dan informasi di dunia digital."},
    {"question": "Apakah kamu lebih tertarik pada teknologi yang berhubungan dengan AI atau Cybersecurity?", "answer": "Saat ini saya lebih tertarik pada AI, tapi Cybersecurity juga sangat penting dan menarik bagi saya."},
    {"question": "Menurut kamu, apa dampak positif dari perkembangan teknologi dalam kehidupan kita?", "answer": "Perkembangan teknologi memberikan kemudahan dalam berbagai aspek kehidupan, mulai dari komunikasi, pendidikan, hingga pekerjaan, dan dapat meningkatkan efisiensi serta produktivitas."},

    # Pengalaman dan Tujuan Masa Depan
    {"question": "Apa tujuan kamu setelah lulus kuliah?", "answer": "Setelah lulus, saya ingin berkontribusi di bidang teknologi untuk mempermudah kehidupan manusia dengan solusi berbasis teknologi."},
    {"question": "Cita-cita kamu apa setelah lulus dari Teknologi Informasi?", "answer": "Cita-cita saya adalah menjadi seorang ahli di bidang AI atau Cybersecurity, dan bisa bekerja di perusahaan teknologi terkemuka."},
    {"question": "Apa yang ingin kamu capai dalam karier teknologi?", "answer": "Saya ingin menjadi seorang profesional yang dapat menciptakan solusi teknologi yang bermanfaat dan berkontribusi besar dalam perkembangan teknologi."},
    {"question": "Apa harapan kamu di masa depan terkait dengan kontribusimu dalam teknologi?", "answer": "Harapan saya adalah dapat menciptakan teknologi yang dapat mempermudah hidup manusia dan memberikan dampak positif bagi masyarakat."},
    {"question": "Apa yang ingin kamu lakukan untuk membantu mempermudah kehidupan manusia lewat teknologi?", "answer": "Saya ingin menciptakan aplikasi dan sistem berbasis teknologi yang bisa memecahkan masalah sehari-hari, seperti dalam hal pendidikan, kesehatan, dan keamanan."},

    # Proyek dan Pengalaman
    {"question": "Pernah ikut lomba atau kompetisi di bidang teknologi?", "answer": "Saya pernah mengikuti Olimpiade Sains Nasional (OSN) tingkat kota dua kali, meskipun belum berhasil meraih juara."},
    {"question": "Apa pengalaman menarik yang kamu dapatkan dari mengikuti OSN?", "answer": "Pengalaman yang saya dapatkan dari OSN adalah belajar lebih banyak tentang persaingan dan bagaimana cara meningkatkan kemampuan diri."},
    {"question": "Walaupun belum berhasil di OSN, apa pelajaran yang kamu ambil dari pengalaman itu?", "answer": "Pelajaran yang saya ambil adalah pentingnya persiapan dan semangat untuk terus belajar, meskipun menghadapi kegagalan."},
    {"question": "Apa pengalaman proyek yang paling berkesan selama kuliah?", "answer": "Pengalaman yang paling berkesan adalah ketika saya mulai mengerjakan proyek pemrograman pertama saya di kampus dan merasa sangat tertantang."},

    # Asal
    {"question": "Kamu asalnya dari mana?", "answer": "Saya asalnya dari Solo"}
]

# Konversi ke dataframe
df_qa = pd.DataFrame(qa_data)

# TF-IDF Vectorizer untuk menghitung kemiripan teks
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df_qa["question"])

# Fungsi chatbot dengan threshold
def chatbot_response(user_question, threshold=0.3):  
    # Transform pertanyaan pengguna ke vektor
    user_vector = vectorizer.transform([user_question])

    # Hitung cosine similarity
    similarity_scores = cosine_similarity(user_vector, question_vectors)

    # Temukan index jawaban dengan similarity tertinggi
    best_match_index = similarity_scores.argmax()
    best_match_score = similarity_scores.max()

    # Logika threshold
    if best_match_score < threshold:
        return "Saya ngga paham."

    return df_qa.loc[best_match_index, "answer"]



# Test chatbot
while True:
    user_input = recognize_speech()
    if user_input.lower() == "exit":
        print("Fahmi : Terima kasih! Sampai jumpa.")
        text = "Terima kasih! Sampai jumpa."
        text_to_speech(text)

        # Return jawaban terbaik
        baris()
        print("         Sesi Perkenalan Telah Selesai")
        baris()
        break

    response = chatbot_response(user_input, threshold=0.3)  
    print(f"Fahmi : {response}")
    text_to_speech(response)
