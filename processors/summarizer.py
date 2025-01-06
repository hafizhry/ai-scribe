from langchain_openai import ChatOpenAI
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class TemplateType(Enum):
    SOAP = "soap"
    APSO = "apso"
    INTAKE = "intake"
    MULTIPLE_SECTION = "multiple_section"
    CUSTOM = "custom"

@dataclass
class Template:
    name: str
    description: str
    prompt: str

class SummaryTemplates:
    @staticmethod
    def get_intake_template() -> Template:
        return Template(
            name="Intake",
            description="Comprehensive initial assessment for first-time visits",
            prompt="""
            Buatkan ringkasan intake assessment dari percakapan berikut dengan format:

            RIWAYAT PASIEN:
            * Informasi Demografis: Usia, status, pekerjaan (jika disebutkan)
            * Riwayat Medis: Kondisi kesehatan yang ada/sebelumnya
            * Riwayat Psikiatris: Pengobatan atau terapi sebelumnya
            * Riwayat Keluarga: Riwayat kesehatan mental dalam keluarga
            
            ASESMEN SAAT INI:
            * Keluhan Utama: Alasan utama mencari bantuan
            * Gejala yang Dilaporkan: Daftar dan deskripsi gejala
            * Pemicu: Faktor-faktor yang memicu gejala
            * Dampak: Bagaimana gejala memengaruhi kehidupan sehari-hari
            
            PENILAIAN RISIKO:
            * Risiko Bunuh Diri: Indikasi pikiran atau rencana
            * Risiko terhadap Orang Lain: Potensi bahaya ke orang lain
            * Faktor Pelindung: Dukungan sosial, coping skills
            
            DIAGNOSIS AWAL:
            * Diagnosis Kerja: Diagnosis sementara berdasarkan DSM-5
            * Diagnosis Banding: Kemungkinan diagnosis lain
            
            RENCANA PENGOBATAN:
            * Rekomendasi Pengobatan: Saran terapi atau obat
            * Tujuan Pengobatan: Target spesifik yang ingin dicapai
            * Langkah Selanjutnya: Rencana tindak lanjut

            Gunakan percakapan berikut untuk membuat ringkasan:
            {text}
            """
        )

    @staticmethod
    def get_soap_template() -> Template:
        return Template(
            name="SOAP",
            description="Structured progress documentation for follow-up visits",
            prompt="""
            Buatkan catatan SOAP (Subjective, Objective, Assessment, Plan) dari percakapan berikut:

            SUBJECTIVE:
            * Keluhan utama pasien
            * Perubahan sejak kunjungan terakhir
            * Efek pengobatan saat ini
            * Gejala baru yang dilaporkan
            
            OBJECTIVE:
            * Penampilan dan perilaku
            * Mood dan afek
            * Proses dan konten pikiran
            * Observasi klinis lainnya
            
            ASSESSMENT:
            * Status kondisi saat ini
            * Respons terhadap pengobatan
            * Perubahan diagnosis (jika ada)
            * Faktor-faktor yang mempengaruhi
            
            PLAN:
            * Modifikasi pengobatan
            * Intervensi terapeutik
            * Rujukan yang diperlukan
            * Jadwal tindak lanjut

            Gunakan percakapan berikut untuk membuat catatan SOAP:
            {text}
            """
        )

    @staticmethod
    def get_apso_template() -> Template:
        return Template(
            name="APSO",
            description="Assessment & Plan focused format with supporting information",
            prompt="""
            Buatkan catatan APSO (Assessment, Plan, Subjective, Objective) dari percakapan berikut:

            ASSESSMENT:
            * Evaluasi status kondisi saat ini
            * Perkembangan sejak kunjungan terakhir
            * Komplikasi atau masalah baru
            * Diagnosis kerja yang diperbarui
            
            PLAN:
            * Rencana pengobatan
            * Modifikasi terapi
            * Pemeriksaan lanjutan
            * Target dan tujuan
            
            SUBJECTIVE:
            * Keluhan utama
            * Gejala yang dilaporkan
            * Perubahan yang diamati pasien
            * Masalah tambahan
            
            OBJECTIVE:
            * Temuan klinis
            * Status mental
            * Hasil evaluasi
            * Observasi perilaku

            Gunakan percakapan berikut untuk membuat catatan APSO:
            {text}
            """
        )

    @staticmethod
    def get_multiple_section_template() -> Template:
        return Template(
            name="Multiple Section",
            description="Detailed sectional analysis of patient's condition",
            prompt="""
            Buatkan analisis terperinci dari percakapan berikut dengan format:

            MASALAH UTAMA:
            * Keluhan utama pasien
            * Gejala primer
            
            DURASI:
            * Waktu mulai gejala
            * Pola perkembangan
            
            TINGKAT KEPARAHAN:
            * Intensitas gejala
            * Perubahan seiring waktu
            
            LOKASI:
            * Area yang terdampak
            * Pola penyebaran (jika ada)
            
            GEJALA TERKAIT:
            * Gejala sekunder
            * Manifestasi lain
            
            FAKTOR YANG MEMPENGARUHI:
            * Faktor pemicu
            * Faktor yang meringankan
            
            PENGOBATAN SEBELUMNYA:
            * Terapi yang pernah dicoba
            * Efektivitas pengobatan

            Gunakan percakapan berikut untuk membuat analisis:
            {text}
            """
        )

    @staticmethod
    def get_custom_template() -> Template:
        # Default custom template (will be replaced when create_custom_template is called)
        return Template(
            name="Custom",
            description="Custom template with user-defined sections",
            prompt="""
            Buatkan ringkasan terstruktur dari percakapan berikut dengan format yang
            akan ditentukan oleh pengguna:

            {text}
            """
        )

    @staticmethod
    def create_custom_template(sections: List[str]) -> Template:
        # Create prompt from custom sections
        prompt_sections = []
        for section in sections:
            prompt_sections.append(f"{section.upper()}:\n* Temuan dan analisis untuk {section}")
        
        prompt = """
        Buatkan ringkasan terstruktur dari percakapan berikut dengan format:

        {sections}

        Gunakan percakapan berikut untuk membuat ringkasan:
        {{text}}
        """.format(sections="\n\n".join(prompt_sections))

        return Template(
            name="Custom",
            description="Custom template with user-defined sections",
            prompt=prompt
        )

class Summarize:
    def __init__(self, api_key: str):
        self.models = {
            "gpt": ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
        }
        self.current_model = self.models["gpt"]
        self.templates = {
            TemplateType.INTAKE: SummaryTemplates.get_intake_template(),
            TemplateType.SOAP: SummaryTemplates.get_soap_template(),
            TemplateType.APSO: SummaryTemplates.get_apso_template(),
            TemplateType.MULTIPLE_SECTION: SummaryTemplates.get_multiple_section_template(),
            TemplateType.CUSTOM: SummaryTemplates.get_custom_template(),  # Initialize with default
        }

    def create_custom_template(self, sections: List[str]) -> None:
        """Create a custom template with specified sections."""
        self.templates[TemplateType.CUSTOM] = SummaryTemplates.create_custom_template(sections)

    def get_available_templates(self) -> Dict[str, str]:
        """Get available template names and their descriptions."""
        return {template_type.value: template.description 
                for template_type, template in self.templates.items()}

    def summarize(self, text: str, template_type: TemplateType) -> str:
        """
        Generate a summary using the specified template.
        
        Args:
            text: The conversation text to summarize
            template_type: The type of template to use
            
        Returns:
            str: The generated summary
        """
        if template_type not in self.templates:
            raise ValueError(f"Template type {template_type} not found")

        template = self.templates[template_type]
        messages = [
            {"role": "system", "content": "You are a medical scribe assistant skilled in creating structured medical documentation in Indonesian language. IF THERE IS NO CONVERSATION, PLEASE WRITE 'No conversation found.'"},
            {"role": "user", "content": template.prompt.format(text=text)}
        ]

        ai_msg = self.current_model.invoke(messages)
        return ai_msg.content if hasattr(ai_msg, 'content') else ai_msg