import streamlit as st
import json
import requests
from typing import Dict, List
import google.generativeai as genai

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = []

    def log_action(self, action: str):
        self.log.append(f"{self.name}: {action}")

class WebBrowserTools:
    def __init__(self):
        self.gemini = genai.GenerativeModel(model_name="gemini-1.5-pro")

    def scrape_company_info(self, company_name: str) -> Dict:
        try:
            company_name_formatted = company_name.replace(' ', '_')
            response = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name_formatted}"
            )
            if response.status_code == 200:
                data = response.json()
                description = data.get("extract", "")
                if not description or "may refer to:" in description:
                    description = self.generate_description_with_gemini(company_name)
            else:
                description = self.generate_description_with_gemini(company_name)
        except Exception:
            description = self.generate_description_with_gemini(company_name)

        return {
            "description": description,
            "products": ["Product A", "Product B"],
            "focus_areas": ["Focus Area 1", "Focus Area 2"]
        }

    def generate_description_with_gemini(self, company_name: str) -> str:
        prompt = f"Please write a 2-3 line professional description about the company '{company_name}'."
        response = self.gemini.generate_content(prompt)
        return response.text.strip()

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Research Agent")
        self.browser_tools = WebBrowserTools()

    def research_company(self, company_name: str) -> Dict:
        self.log_action(f"Researching {company_name}")
        company_data = self.browser_tools.scrape_company_info(company_name)
        industry = self._classify_industry(company_data["description"])
        return {
            "company": company_name,
            "industry": industry,
            "offerings": company_data["products"],
            "strategic_focus": company_data["focus_areas"],
            "description": company_data["description"]
        }

    def _classify_industry(self, description: str) -> str:
        description = description.lower()
        if "automotive" in description or "vehicle" in description:
            return "Automotive"
        elif "finance" in description or "bank" in description:
            return "Finance"
        elif "e-commerce" in description or "retail" in description:
            return "E-commerce/Retail"
        elif "technology" in description or "software" in description:
            return "Technology"
        elif "entertainment" in description or "media" in description:
            return "Entertainment/Media"
        elif "healthcare" in description or "medical" in description:
            return "Healthcare"
        else:
            return "General Industry"

class MarketAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Market Analysis Agent")
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    def generate_use_cases(self, industry_data: Dict) -> List[Dict]:
        self.log_action("Generating AI/ML/GenAI use cases")
        prompt = f"""You are an AI business consultant.
Analyze the following industry and suggest AI/ML and Generative AI (GenAI) use cases:

Industry: {industry_data['industry']}
Description: {industry_data['description']}
Offerings: {industry_data['offerings']}
Strategic Focus Areas: {industry_data['strategic_focus']}

Please suggest:
- Practical AI/ML/GenAI solutions
- Improvements to customer experience, operations, supply chain
- Internal GenAI solutions (chatbots, automated reporting, document search)

Respond with a JSON array format. Each item should have fields: "use_case", "description", and "feasibility".
"""
        response = self.model.generate_content(prompt)
        return self._parse_response(response.text)

    def _parse_response(self, response_content: str) -> List[Dict]:
        try:
            return json.loads(response_content)
        except Exception:
            return [{"use_case_summary": response_content}]

class ResourceCollectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Resource Collector Agent")

    def find_resources(self, use_cases: List[Dict]) -> Dict:
        self.log_action("Finding resources")
        resources = {}
        for case in use_cases:
            title = case.get("use_case", case.get("use_case_summary", "general"))
            resources[title] = {
                "datasets": [f"https://kaggle.com/search?q={title.replace(' ', '+')}"],
                "models": [f"https://huggingface.co/models?search={title.replace(' ', '+')}"],
                "github_projects": [f"https://github.com/search?q={title.replace(' ', '+')}"]
            }
        return resources

class MultiAgentSystem:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.market_agent = MarketAnalysisAgent()
        self.resource_agent = ResourceCollectorAgent()

    def run(self, company_name: str) -> Dict:
        company_info = self.research_agent.research_company(company_name)
        use_cases = self.market_agent.generate_use_cases(company_info)
        resources = self.resource_agent.find_resources(use_cases)
        return {
            "company_info": company_info,
            "ai_use_cases": use_cases,
            "resource_assets": resources
        }

st.set_page_config(page_title="Multi-Agent AI Use Case Generator", page_icon="🤖")
st.title("🤖 Multi-Agent AI Use Case Generator")
st.write("Enter a company name to generate AI/ML/GenAI use cases and resources:")

company_name = st.text_input("Company Name")

if st.button("Run Agents 🚀"):
    if company_name:
        orchestrator = MultiAgentSystem()
        with st.spinner('Agents are working...'):
            results = orchestrator.run(company_name)
        st.success('✅ Completed!')

        company_info = results['company_info']
        use_cases = results['ai_use_cases']
        resources = results['resource_assets']

        st.header("📄 Company Information")
        st.markdown(f"**Company:** {company_info['company']}")
        st.markdown(f"**Industry:** {company_info['industry']}")
        st.markdown(f"**Description:** {company_info['description']}")
        st.markdown(f"**Offerings:**")
        for item in company_info['offerings']:
            st.markdown(f"- {item}")
        st.markdown(f"**Strategic Focus Areas:**")
        for area in company_info['strategic_focus']:
            st.markdown(f"- {area}")

        st.header("🚀 AI/GenAI Use Cases")
        for idx, case in enumerate(use_cases, 1):
            st.subheader(f"{idx}. {case.get('use_case', 'Unnamed Use Case')}")
            st.markdown(f"**Description:** {case.get('description', 'No description')}")
            st.markdown(f"**Feasibility:** {case.get('feasibility', 'Unknown')}")
            st.markdown("---")

        st.header("📚 Resources (Datasets, Models, GitHub Projects)")
        for title, links in resources.items():
            st.subheader(f"🔹 {title}")
            st.markdown("**Datasets:**")
            for link in links["datasets"]:
                st.markdown(f"- [{link}]({link})")
            st.markdown("**Models:**")
            for link in links["models"]:
                st.markdown(f"- [{link}]({link})")
            st.markdown("**GitHub Projects:**")
            for link in links["github_projects"]:
                st.markdown(f"- [{link}]({link})")
            st.markdown("---")

        if st.checkbox("Download results as JSON"):
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(results, indent=2),
                file_name=f"{company_name}_analysis.json",
                mime="application/json"
            )
    else:
        st.warning("Please enter a company name first!")
