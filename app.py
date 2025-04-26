
import streamlit as st
import json
import requests
from typing import Dict, List
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log = []
    def log_action(self, action: str):
        self.log.append(f"{self.name}: {action}")

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
            description = f"Error fetching data: {str(e)}"
        return {
            "description": description,
            "products": ["Product A", "Product B"],
            "focus_areas": ["Focus Area 1", "Focus Area 2"]
        }

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
        prompt = f"""You are an AI consultant. Given the following industry details:
Industry: {industry_data['industry']}
Description: {industry_data['description']}
Offerings: {industry_data['offerings']}
Strategic Focus: {industry_data['strategic_focus']}
Suggest AI/ML and GenAI use cases. Respond in JSON array with "use_case", "description", and "feasibility".
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

st.title("🔎 Multi-Agent Company Analyzer")
company = st.text_input("Enter Company Name:", "")

if st.button("Analyze"):
    if company:
        orchestrator = MultiAgentSystem()
        results = orchestrator.run(company)
        st.subheader("Company Information")
        st.json(results["company_info"])
        st.subheader("AI Use Cases")
        st.json(results["ai_use_cases"])
        st.subheader("Resources")
        st.json(results["resource_assets"])
    else:
        st.warning("Please enter a company name!")
