import json
import requests
import streamlit as st
from typing import Dict, List
import google.generativeai as genai

genai.configure(api_key="YOUR-GEMINI-API-KEY-HERE")

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = []

    def log_action(self, action: str):
        self.log.append(f"{self.name}: {action}")

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Research Agent")
        self.browser_tools = WebBrowserTools()

    def research_company(self, company_name: str) -> Dict:
        self.log_action(f"Researching company: {company_name}")
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
        self.log_action("Generating AI/GenAI/ML use cases")
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
        self.dataset_sources = ["Kaggle", "HuggingFace", "GitHub"]

    def find_resources(self, use_cases: List[Dict]) -> Dict:
        self.log_action("Finding resources for use cases")
        resources = {}
        for case in use_cases:
            title = case.get("use_case", case.get("use_case_summary", "general"))
            resources[title] = {
                "datasets": self._search_kaggle(title),
                "models": self._search_huggingface(title),
                "github_projects": self._search_github(title)
            }
        return resources

    def _search_kaggle(self, query: str) -> List[str]:
        return [f"https://kaggle.com/search?q={query.replace(' ', '+')}"]

    def _search_huggingface(self, query: str) -> List[str]:
        return [f"https://huggingface.co/models?search={query.replace(' ', '+')}"]

    def _search_github(self, query: str) -> List[str]:
        return [f"https://github.com/search?q={query.replace(' ', '+')}"]

class WebBrowserTools:
    def scrape_company_info(self, company_name: str) -> Dict:
        try:
            response = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{company_name}"
            )
            if response.status_code == 200:
                data = response.json()
                description = data.get("extract", f"No information found for {company_name}.")
            else:
                description = f"No information found for {company_name}."
        except Exception as e:
            description = f"Error fetching data for {company_name}: {str(e)}"
        return {
            "description": description,
            "products": ["Product A", "Product B"],
            "focus_areas": ["Focus Area 1", "Focus Area 2"]
        }

class MultiAgentSystem:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.market_agent = MarketAnalysisAgent()
        self.resource_agent = ResourceCollectorAgent()

    def run(self, company_name: str) -> Dict:
        company_info = self.research_agent.research_company(company_name)
        use_cases = self.market_agent.generate_use_cases(company_info)
        resources = self.resource_agent.find_resources(use_cases)
        final_output = {
            "company_info": company_info,
            "ai_use_cases": use_cases,
            "resource_assets": resources
        }
        return final_output

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

        st.subheader("📄 Company Info")
        st.json(results['company_info'])

        st.subheader("🚀 AI/GenAI Use Cases")
        st.json(results['ai_use_cases'])

        st.subheader("📚 Resources (Datasets, Models, GitHub)")
        st.json(results['resource_assets'])

        save_option = st.checkbox("Download result as JSON?")
        if save_option:
            st.download_button(
                label="Download JSON",
                data=json.dumps(results, indent=2),
                file_name=f"{company_name}_ai_use_cases.json",
                mime="application/json"
            )
    else:
        st.warning("Please enter a company name first!")
