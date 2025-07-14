# app/data_quality/multi_table_lotus.py
# Enhanced version that supports multiple tables and audit/compliance queries
import pandas as pd
import os
import sys
import json
import requests
from typing import List, Tuple, Union, Dict, Any

class MultiTableLotus:
    """Enhanced semantic processing with multi-table support for audit and compliance queries"""
    
    def __init__(self, dfs, model="mistral", backend="ollama", mode="query", api_key=None):
        self.model = model
        self.backend = backend
        self.mode = mode
        self.api_key = api_key
        
        # Handle different input formats and store all tables
        if isinstance(dfs, list) and len(dfs) > 0 and isinstance(dfs[0], tuple):
            self.tables = {name: df for name, df in dfs}
        elif isinstance(dfs, pd.DataFrame):
            self.tables = {"default": dfs}
        elif isinstance(dfs, dict):
            self.tables = dfs
        else:
            raise ValueError("Invalid input: must be list of (name, df), single DataFrame, or dict")
    
    def _call_ollama_api(self, prompt: str) -> str:
        """Call Ollama API directly"""
        try:
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 16384  # Even larger context for multi-table
                }
            }
            
            response = requests.post(url, json=payload, timeout=600)  # Longer timeout
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def _call_huggingface_api(self, prompt: str) -> str:
        """Call HuggingFace API directly"""
        try:
            url = f"https://api-inference.huggingface.co/models/{self.model}"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1024,  # More tokens for complex queries
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            return str(result)
            
        except Exception as e:
            raise Exception(f"HuggingFace API error: {e}")
    
    def _call_google_api(self, prompt: str) -> str:
        """Call Google API directly"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            
            response = model.generate_content(prompt)
            return response.text.strip()
            
        except ImportError:
            raise Exception("Google Generative AI not installed. Run: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Google API error: {e}")
    
    def _get_llm_response(self, prompt: str) -> str:
        """Get LLM response based on backend"""
        if self.backend == "ollama":
            return self._call_ollama_api(prompt)
        elif self.backend == "huggingface":
            if not self.api_key:
                raise Exception("API key required for HuggingFace backend")
            return self._call_huggingface_api(prompt)
        elif self.backend == "google":
            if not self.api_key:
                raise Exception("API key required for Google backend")
            return self._call_google_api(prompt)
        else:
            raise Exception(f"Unsupported backend: {self.backend}")
    
    def _analyze_tables_for_relationships(self) -> Dict[str, Any]:
        """Analyze tables to identify potential relationships and join keys"""
        table_info = {}
        potential_joins = []
        
        for name, df in self.tables.items():
            # Analyze each table
            columns = df.columns.tolist()
            dtypes = {col: str(df[col].dtype) for col in columns}
            
            # Identify potential key columns
            key_candidates = []
            for col in columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'number']):
                    key_candidates.append(col)
                elif df[col].nunique() == len(df):  # Unique values
                    key_candidates.append(col)
            
            # Identify name/text columns for conflict detection
            name_columns = []
            for col in columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['name', 'company', 'vendor', 'customer', 'employee']):
                    name_columns.append(col)
                elif df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.8:
                    name_columns.append(col)
            
            table_info[name] = {
                'columns': columns,
                'dtypes': dtypes,
                'key_candidates': key_candidates,
                'name_columns': name_columns,
                'shape': df.shape
            }
        
        # Find potential joins between tables
        table_names = list(self.tables.keys())
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                common_columns = set(table_info[table1]['columns']) & set(table_info[table2]['columns'])
                if common_columns:
                    potential_joins.append({
                        'table1': table1,
                        'table2': table2,
                        'common_columns': list(common_columns)
                    })
        
        return {
            'table_info': table_info,
            'potential_joins': potential_joins
        }
    
    def _create_multi_table_prompt(self, question: str) -> str:
        """Create enhanced prompt for multi-table analysis including audit and compliance queries"""
        
        # Analyze table relationships
        analysis = self._analyze_tables_for_relationships()
        table_info = analysis['table_info']
        potential_joins = analysis['potential_joins']
        
        # Build comprehensive table description
        tables_desc = []
        for name, df in self.tables.items():
            info = table_info[name]
            sample_data = df.head(2).to_string(index=False) if len(df) > 0 else "No data"
            
            tables_desc.append(f"""
Table: {name}
Shape: {info['shape'][0]} rows, {info['shape'][1]} columns
Columns: {', '.join(info['columns'])}
Key Candidates: {', '.join(info['key_candidates']) if info['key_candidates'] else 'None identified'}
Name/Text Columns: {', '.join(info['name_columns']) if info['name_columns'] else 'None identified'}
Sample Data:
{sample_data}
""")
        
        # Build join information
        joins_desc = ""
        if potential_joins:
            joins_desc = "\nPotential Table Relationships:\n"
            for join in potential_joins:
                joins_desc += f"- {join['table1']} ↔ {join['table2']} via: {', '.join(join['common_columns'])}\n"
        
        # Create specialized prompts for different query types
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['conflict', 'duplicate', 'same', 'both', 'also']):
            # Conflict of Interest Detection
            prompt = f"""You are an expert compliance analyst specializing in conflict of interest detection.

Available Tables:
{chr(10).join(tables_desc)}
{joins_desc}

User Question: "{question}"

Your task: Generate Python pandas code to detect potential conflicts of interest across tables.

🔍 Conflict Detection Strategies:
1. **Name Matching**: Find similar names across different tables (employees vs vendors, customers vs suppliers)
2. **Email Domain Matching**: Identify shared email domains between parties
3. **Address Matching**: Find shared addresses or locations
4. **ID Cross-Reference**: Check if same IDs appear in different contexts
5. **Relationship Detection**: Identify potential undisclosed relationships

💻 Code Generation Rules:
1. Available DataFrames: {', '.join([f"{name} as {name.lower().replace(' ', '_').replace('-', '_')}" for name in self.tables.keys()])}
2. Return ONLY executable Python code
3. Use fuzzy matching techniques (.str.contains with regex)
4. Handle case sensitivity with case=False
5. Create summary results showing potential conflicts
6. Use meaningful column names in results

🎯 Example Patterns:
- Name conflicts: Check for similar names across employee and vendor tables
- Email conflicts: Find shared domains between customers and suppliers
- Address conflicts: Identify same addresses for different entity types

Generate the pandas code to detect conflicts:"""

        elif any(word in question_lower for word in ['audit', 'compliance', 'rule', 'violation', 'missing', 'unauthorized', 'approval']):
            # Audit Rules and Compliance
            prompt = f"""You are an expert auditor specializing in compliance rule detection.

Available Tables:
{chr(10).join(tables_desc)}
{joins_desc}

User Question: "{question}"

Your task: Generate Python pandas code to identify audit rule violations and compliance issues.

🔍 Audit Rule Categories:
1. **Authorization Rules**: Transactions without proper approval
2. **Threshold Rules**: Values exceeding defined limits
3. **Completeness Rules**: Missing required fields or documents
4. **Segregation Rules**: Same person in multiple conflicting roles
5. **Timing Rules**: Transactions outside allowed periods
6. **Duplicate Rules**: Duplicate entries that shouldn't exist

💻 Code Generation Rules:
1. Available DataFrames: {', '.join([f"{name} as {name.lower().replace(' ', '_').replace('-', '_')}" for name in self.tables.keys()])}
2. Return ONLY executable Python code
3. Use boolean indexing for rule violations
4. Create clear violation indicators
5. Include violation counts and percentages
6. Sort by severity or impact

🎯 Example Patterns:
- Missing approvals: transactions.isnull().any(axis=1)
- Over limits: df[df['amount'] > df['limit']]
- Unauthorized: df[~df['approver'].isin(authorized_list)]

Generate the pandas code to detect violations:"""

        elif any(word in question_lower for word in ['join', 'merge', 'combine', 'across', 'between']):
            # Multi-table joins
            prompt = f"""You are an expert data analyst specializing in multi-table analysis.

Available Tables:
{chr(10).join(tables_desc)}
{joins_desc}

User Question: "{question}"

Your task: Generate Python pandas code to join and analyze data across multiple tables.

🔗 Join Strategies:
1. **Inner Join**: Records that exist in both tables
2. **Left Join**: All records from first table, matching from second
3. **Outer Join**: All records from both tables
4. **Key Detection**: Automatically identify join columns

💻 Code Generation Rules:
1. Available DataFrames: {', '.join([f"{name} as {name.lower().replace(' ', '_').replace('-', '_')}" for name in self.tables.keys()])}
2. Return ONLY executable Python code
3. Use pd.merge() for joining tables
4. Handle missing values in joins appropriately
5. Create meaningful result with clear column names
6. Limit final results to top 50 rows for readability

🎯 Join Examples:
- pd.merge(table1, table2, on='common_id', how='inner')
- pd.merge(employees, departments, left_on='dept_id', right_on='id')

Generate the pandas code for multi-table analysis:"""

        else:
            # General multi-table query
            prompt = f"""You are an expert data analyst with access to multiple related tables.

Available Tables:
{chr(10).join(tables_desc)}
{joins_desc}

User Question: "{question}"

Your task: Generate Python pandas code to answer the question using appropriate tables.

💡 Multi-Table Intelligence:
1. Identify which tables are most relevant to the question
2. Use joins when relationships are needed
3. Focus on single table if question is specific
4. Combine insights from multiple tables when beneficial

💻 Code Generation Rules:
1. Available DataFrames: {', '.join([f"{name} as {name.lower().replace(' ', '_').replace('-', '_')}" for name in self.tables.keys()])}
2. Return ONLY executable Python code
3. Choose the most appropriate table(s) for the question
4. Use joins only when necessary
5. Create clear, interpretable results

Generate the pandas code:"""
        
        return prompt
    
    def run_multi_table_query(self, question: str) -> Dict[str, Any]:
        """Run multi-table semantic query with support for audit and compliance"""
        try:
            # Create multi-table prompt
            prompt = self._create_multi_table_prompt(question)
            
            # Get response from LLM
            code_response = self._get_llm_response(prompt)
            
            # Clean up the code response
            if "```python" in code_response:
                code_response = code_response.split("```python")[1].split("```")[0].strip()
            elif "```" in code_response:
                parts = code_response.split("```")
                if len(parts) >= 3:
                    code_response = parts[1].strip()
                else:
                    code_response = max(parts, key=len).strip()
            
            # Execute the generated code
            try:
                # Create execution environment with all tables
                exec_globals = {
                    'pd': pd,
                    '__builtins__': __builtins__
                }
                
                # Add all tables to the environment with cleaned names
                for name, df in self.tables.items():
                    clean_name = name.lower().replace(' ', '_').replace('-', '_')
                    exec_globals[clean_name] = df
                
                exec(code_response, exec_globals)
                
                # Try to get the result
                if 'result' in exec_globals:
                    result = exec_globals['result']
                else:
                    # Evaluate the last meaningful line
                    lines = [line.strip() for line in code_response.strip().split('\n') if line.strip()]
                    if lines:
                        last_line = lines[-1]
                        if not last_line.startswith(('import ', 'from ', '#')):
                            if last_line.startswith('print('):
                                last_line = last_line[6:-1]
                            result = eval(last_line, exec_globals)
                        else:
                            result = "Multi-table analysis completed"
                    else:
                        result = "Multi-table analysis completed"
                
                # Process the result
                if isinstance(result, pd.DataFrame):
                    if len(result) == 0:
                        return {"type": "text", "content": f"🔍 Multi-table analysis found no results for: {question}"}
                    else:
                        return {"type": "dataframe", "content": result.head(100)}  # More rows for complex analysis
                
                elif isinstance(result, pd.Series):
                    if len(result) == 0:
                        return {"type": "text", "content": f"🔍 Multi-table analysis found no results for: {question}"}
                    else:
                        result_df = result.reset_index()
                        return {"type": "dataframe", "content": result_df.head(50)}
                
                elif isinstance(result, (int, float)):
                    return {"type": "text", "content": f"🎯 Multi-table analysis result: {result}"}
                
                else:
                    return {"type": "text", "content": f"🔍 Multi-table analysis: {str(result)}"}
            
            except Exception as exec_error:
                return {
                    "type": "error",
                    "content": f"Error executing multi-table query: {str(exec_error)}\n\nGenerated code:\n```python\n{code_response}\n```"
                }
                
        except Exception as e:
            return {
                "type": "error",
                "content": f"Multi-table processing error: {str(e)}"
            }
    
    # Compatibility methods
    def run_lotus(self, question: str) -> Dict[str, Any]:
        return self.run_multi_table_query(question)
    
    def run_semantic_query(self, question: str) -> Dict[str, Any]:
        return self.run_multi_table_query(question)

# Update the main LotusLLM class to use multi-table capabilities
LotusLLM = MultiTableLotus