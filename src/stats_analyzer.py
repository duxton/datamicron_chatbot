import pandas as pd
from typing import Dict, Any, Optional
import config
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class StatsAnalyzer:

    def __init__(self, dataframe: pd.DataFrame):

        self.df = dataframe
        self.llm = None
        self.setup_llm()
        
    def setup_llm(self):
        if config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=0.0,  # Zero temperature for code generation
                openai_api_key=config.OPENAI_API_KEY,
                max_tokens=1024
            )
            print("✓ Statistics Analyzer initialized with OpenAI")
        else:
            print("⚠ Warning: OPENAI_API_KEY not set")
    
    def get_dataset_info(self) -> str:
        info = f"""Dataset Information:
            - Total rows: {len(self.df)}
            - Columns: {', '.join(self.df.columns.tolist())}

            Key columns:
            - title: Title of the news article
            - article_content: Full content of the article
            - summary: Summary of the article
            - author: Source/author of the news
            - sentiment: Sentiment (Positive/Negative/Neutral)
            - timestamp: Date and time of the article
            - url: URL of the article
            - source_country: Country of the source

            Sample data types:
            {self.df.dtypes.to_string()}
            """
        return info
    
    def generate_analysis_code(self, query: str) -> Dict[str, Any]:
        if not self.llm:
            return {
                'success': False,
                'error': 'OpenAI API not configured',
                'code': '',
                'explanation': ''
            }
        
        dataset_info = self.get_dataset_info()
        
        template = """You are a data analysis expert. Generate Python/Pandas code to answer the user's question about a news dataset.

            {dataset_info}

            User Question: {question}

            Instructions:
            1. Generate ONLY the Python/Pandas code needed to answer the question
            2. Assume the DataFrame is available as 'df'
            3. The code should assign the result to a variable called 'result'
            4. Use proper Pandas methods (value_counts, groupby, filtering, etc.)
            5. Handle date/time parsing if needed (timestamp column format: 'DD-MM-YY HH:MM')
            6. Keep the code safe and simple (no file operations, no imports, no system calls)
            7. Return ONLY valid Python code, no explanations in the code

            Also provide a brief explanation of what the code does.

            Return your response in this exact JSON format:
            {{
                "code": "your python code here",
                "explanation": "brief explanation of what the code does"
            }}

            Example for "How many positive news are there?":
            {{
                "code": "result = df[df['sentiment'] == 'Positive'].shape[0]",
                "explanation": "Counts the number of rows where sentiment is 'Positive'"
            }}

            Now generate the code for the user's question."""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            response_text = chain.invoke({
                "dataset_info": dataset_info,
                "question": query
            })
            response_text = response_text.strip()
            if '```json' in response_text:
                response_text = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if response_text:
                    response_text = response_text.group(1)
            elif '```' in response_text:
                response_text = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if response_text:
                    response_text = response_text.group(1)

            parsed = json.loads(response_text)
            
            return {
                'success': True,
                'code': parsed.get('code', ''),
                'explanation': parsed.get('explanation', ''),
                'error': None
            }
            
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing JSON response: {e}")
            print(f"Response text: {response_text}")
            return {
                'success': False,
                'error': f'Failed to parse response: {str(e)}',
                'code': '',
                'explanation': ''
            }
        except Exception as e:
            print(f"✗ Error generating analysis code: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': '',
                'explanation': ''
            }
    
    def execute_analysis(self, code: str) -> Dict[str, Any]:
        try:
            safe_globals = {
                'df': self.df,
                'pd': pd,
                'result': None
            }
            exec(code, safe_globals)
            result = safe_globals.get('result')

            if result is None:
                return {
                    'success': False,
                    'error': 'Code did not produce a result',
                    'result': None,
                    'formatted_result': ''
                }
            formatted_result = self.format_result(result)
            
            return {
                'success': True,
                'result': result,
                'formatted_result': formatted_result,
                'error': None
            }
            
        except Exception as e:
            print(f"✗ Error executing analysis code: {e}")
            return {
                'success': False,
                'error': str(e),
                'result': None,
                'formatted_result': ''
            }
    
    def format_result(self, result: Any) -> str:
        if isinstance(result, pd.Series):
            return result.to_string()
        elif isinstance(result, pd.DataFrame):
            return result.to_string()
        elif isinstance(result, (int, float)):
            return str(result)
        elif isinstance(result, dict):
            return json.dumps(result, indent=2)
        elif isinstance(result, list):
            return '\n'.join(str(item) for item in result)
        else:
            return str(result)
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        code_result = self.generate_analysis_code(query)
        
        if not code_result['success']:
            return {
                'success': False,
                'query': query,
                'error': code_result['error'],
                'result': None,
                'formatted_result': '',
                'code': '',
                'explanation': ''
            }
        exec_result = self.execute_analysis(code_result['code'])
        
        return {
            'success': exec_result['success'],
            'query': query,
            'result': exec_result.get('result'),
            'formatted_result': exec_result.get('formatted_result', ''),
            'code': code_result['code'],
            'explanation': code_result['explanation'],
            'error': exec_result.get('error')
        }
    
    def get_quick_stats(self) -> Dict[str, Any]:
        stats = {
            'total_articles': len(self.df),
            'sentiment_distribution': self.df['sentiment'].value_counts().to_dict() if 'sentiment' in self.df.columns else {},
            'top_authors': self.df['author'].value_counts().head(5).to_dict() if 'author' in self.df.columns else {},
            'date_range': {
                'earliest': self.df['timestamp'].min() if 'timestamp' in self.df.columns else 'N/A',
                'latest': self.df['timestamp'].max() if 'timestamp' in self.df.columns else 'N/A'
            }
        }
        return stats
