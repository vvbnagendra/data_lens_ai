# app/data_quality/lotus_llm_adapter.py
# Enhanced multi-table semantic processing (no Lotus-AI dependency required)
import pandas as pd
import os
import sys
import json
import requests
from typing import List, Tuple, Union, Dict, Any
import re

class LotusLLM:
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
        """Create enhanced prompt for multi-table analysis with VERY explicit variable instructions"""
        
        # Analyze table relationships
        analysis = self._analyze_tables_for_relationships()
        table_info = analysis['table_info']
        potential_joins = analysis['potential_joins']
        
        # Build EXACT variable mapping with multiple formats
        dataframe_vars = {}
        available_vars_list = []
        
        for name, df in self.tables.items():
            info = table_info[name]
            # Much more aggressive cleaning for variable names
            clean_var_name = (name.lower()
                            .replace(' ', '_')
                            .replace('-', '_')
                            .replace(':', '_')
                            .replace('.', '_')  # Replace dots with underscores
                            .replace('(', '_')
                            .replace(')', '_')
                            .replace(',', '_'))
            
            # Remove consecutive underscores and trailing underscores
            clean_var_name = '_'.join(filter(None, clean_var_name.split('_')))
            
            # Ensure it starts with a letter or underscore (valid Python variable)
            if clean_var_name[0].isdigit():
                clean_var_name = 'df_' + clean_var_name
            
            dataframe_vars[name] = clean_var_name
            available_vars_list.append(clean_var_name)
            
        # Create VERY explicit variable declaration section
        variable_section = "AVAILABLE DATAFRAME VARIABLES (USE THESE EXACT NAMES):\n"
        for name, df in self.tables.items():
            clean_var_name = dataframe_vars[name]
            variable_section += f"- Variable name: {clean_var_name}\n"
            variable_section += f"  Original file: {name}\n"
            variable_section += f"  Columns: {list(df.columns)}\n"
            variable_section += f"  Shape: {df.shape}\n"
            variable_section += f"  Use like: {clean_var_name}.head()\n\n"
        
        # Build join information with exact variable names
        joins_section = ""
        if potential_joins:
            joins_section = "AVAILABLE JOINS:\n"
            for join in potential_joins:
                var1 = dataframe_vars[join['table1']]
                var2 = dataframe_vars[join['table2']]
                common_cols = join['common_columns']
                joins_section += f"pd.merge({var1}, {var2}, on={common_cols})\n"
        
        # Create UNIVERSAL prompt that works for all query types
        prompt = f"""You are a Python pandas expert. Generate code to answer the user's question.

{variable_section}

{joins_section}

User Question: "{question}"

MANDATORY RULES - FOLLOW EXACTLY:
1. Only use these variable names: {', '.join(available_vars_list)}
2. NEVER use pd.read_csv() - data is already loaded
3. NEVER use .csv syntax - use the variable names directly
4. NEVER use undefined variables like 'employee_master' or 'salary'
5. USE ONLY the exact variable names listed above

CORRECT EXAMPLES:
✅ {available_vars_list[0]}.head()
✅ pd.merge({available_vars_list[0]}, {available_vars_list[1] if len(available_vars_list) > 1 else available_vars_list[0]}, on='common_column')
✅ {available_vars_list[0]}[{available_vars_list[0]}['column'] > value]

WRONG EXAMPLES (DO NOT DO THIS):
❌ employee_master.csv
❌ salary.csv  
❌ pd.read_csv('file.csv')
❌ employee_master (undefined variable)

Your task: Write Python code using ONLY the available variables listed above.
Return ONLY executable Python code with no explanations.
"""
        
        return prompt
    
    def run_lotus(self, question: str) -> Dict[str, Any]:
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
                
                # Add all tables to the environment with cleaned names AND log them
                print(f"DEBUG: Adding DataFrames to execution environment:")
                df_var_mapping = {}  # Track original name to clean name mapping
                
                for name, df in self.tables.items():
                    # Much more aggressive cleaning for variable names
                    clean_name = (name.lower()
                                  .replace(' ', '_')
                                  .replace('-', '_')
                                  .replace(':', '_')
                                  .replace('.', '_')  # Replace dots with underscores
                                  .replace('(', '_')
                                  .replace(')', '_')
                                  .replace(',', '_'))
                    
                    # Remove consecutive underscores and trailing underscores
                    clean_name = '_'.join(filter(None, clean_name.split('_')))
                    
                    # Ensure it starts with a letter or underscore (valid Python variable)
                    if clean_name and clean_name[0].isdigit(): # Added 'clean_name' check
                        clean_name = 'df_' + clean_name
                    elif not clean_name: # Handle case where cleaning might result in empty string
                        clean_name = 'df_unnamed_table' # Or raise an error, depending on desired behavior
                    
                    exec_globals[clean_name] = df
                    df_var_mapping[name] = clean_name
                    print(f"   Added: {clean_name} (original: {name}, shape: {df.shape})")
                
                # Also add matplotlib and plotly if available
                has_matplotlib = False
                has_plotly = False
                try:
                    import matplotlib.pyplot as plt
                    exec_globals['plt'] = plt
                    has_matplotlib = True
                    print(f"   Added: plt (matplotlib)")
                except ImportError:
                    print(f"   Warning: matplotlib not available")
                
                try:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    exec_globals['px'] = px
                    exec_globals['go'] = go
                    has_plotly = True
                    print(f"   Added: px, go (plotly)")
                except ImportError:
                    print(f"   Warning: plotly not available")
                
                print(f"DEBUG: Available variables: {list(exec_globals.keys())}")
                print(f"DEBUG: Variable mapping: {df_var_mapping}")
                print(f"DEBUG: Generated code:")
                print(f"```python\n{code_response}\n```")
                
                # Try to auto-fix common variable name issues
                fixed_code = code_response
                
                # Create comprehensive fix mapping
                fix_mapping = {}
                
                # Add mappings for original names with various formats
                for original_name, clean_var in df_var_mapping.items():
                    # Original name with .csv
                    fix_mapping[original_name] = clean_var
                    
                    # Original name without .csv (if it had .csv initially)
                    if original_name.endswith('.csv'):
                        fix_mapping[original_name.replace('.csv', '')] = clean_var
                    
                    # Original name lowercased (without .csv)
                    fix_mapping[original_name.replace('.csv', '').lower()] = clean_var
                    
                    # Original name lowercased (with .csv)
                    fix_mapping[original_name.lower()] = clean_var

                    # Aggressively cleaned versions LLM might guess
                    aggressive_cleaned_wrong = (original_name.lower()
                                                .replace(' ', '_')
                                                .replace('-', '_')
                                                .replace(':', '_')
                                                .replace('.', '_')
                                                .replace('(', '_')
                                                .replace(')', '_')
                                                .replace(',', '_'))
                    aggressive_cleaned_wrong = '_'.join(filter(None, aggressive_cleaned_wrong.split('_')))
                    if aggressive_cleaned_wrong.startswith('df_') and len(aggressive_cleaned_wrong) > 3 and aggressive_cleaned_wrong[3].isdigit():
                        pass # already prefixed, no change
                    elif aggressive_cleaned_wrong and aggressive_cleaned_wrong[0].isdigit():
                        aggressive_cleaned_wrong = 'df_' + aggressive_cleaned_wrong

                    # Only add if it's genuinely different from the clean_var to avoid recursive issues
                    if aggressive_cleaned_wrong != clean_var:
                        fix_mapping[aggressive_cleaned_wrong] = clean_var


                # Apply fixes in order of longest to shortest to avoid partial replacements
                sorted_fixes = sorted(fix_mapping.items(), key=lambda x: len(x[0]), reverse=True)
                
                import re # Ensure re is imported at the top of the file or here if needed locally
                for wrong_var, correct_var in sorted_fixes:
                    # Use re.sub with word boundaries \b to replace only whole words
                    pattern = r'\b' + re.escape(wrong_var) + r'\b'
                    if re.search(pattern, fixed_code) and wrong_var != correct_var:
                        # Only apply replacement if the actual content is wrong and needs correction
                        # This also prevents self-replacement if wrong_var somehow equals correct_var
                        fixed_code = re.sub(pattern, correct_var, fixed_code)
                        print(f"AUTO-FIX: Replaced '{wrong_var}' with '{correct_var}'")
                
                if fixed_code != code_response:
                    print(f"DEBUG: Fixed code:")
                    print(f"```python\n{fixed_code}\n```")
                    code_response = fixed_code
                
                # --- New Logic for Plotting and Result Handling ---
                is_plotting_code = False
                if has_matplotlib and ('plt.show()' in code_response or 'plt.savefig(' in code_response):
                    is_plotting_code = True
                if has_plotly and ('fig.show()' in code_response or 'fig.write_image(' in code_response):
                    is_plotting_code = True

                # Execute the code
                exec(code_response, exec_globals)
                
                # Check for plotting first
                if is_plotting_code:
                    # You might want to save the plot here if your application supports it
                    # For now, just return a success message
                    return {"type": "text", "content": "Plot generated successfully. Please check your plot viewer or saved file."}

                # Try to get the result from exec_globals (if a variable was assigned)
                result = None
                if 'result' in exec_globals:
                    result = exec_globals['result']
                else:
                    # If 'result' not found, try to evaluate the last line
                    lines = [line.strip() for line in code_response.strip().split('\n') if line.strip()]
                    if lines:
                        last_line = lines[-1]
                        # Avoid evaluating plotting calls or imports as results
                        if not last_line.startswith(('import ', 'from ', '#', 'plt.', 'fig.', 'px.', 'go.')):
                            try:
                                # Use compile and eval for safer execution of a single line
                                # And to catch the value returned by the last expression
                                result = eval(compile(last_line, '<string>', 'eval'), exec_globals)
                            except SyntaxError:
                                # If it's not a valid expression for eval (e.g., a statement), don't force it
                                result = "Code executed, but no direct result captured from the last line."
                            except Exception as e:
                                print(f"DEBUG: Error evaluating last line: {e}")
                                result = f"Code executed, but failed to evaluate the last line as a result: {e}"
                        else:
                            result = "Code executed, but the last line was a statement (e.g., plot, import)."
                    else:
                        result = "No executable lines in the generated code."

                # Process the result
                if isinstance(result, pd.DataFrame):
                    if len(result) == 0:
                        return {"type": "text", "content": f"Multi-table analysis found no results for: {question}"}
                    else:
                        return {"type": "dataframe", "content": result.head(100)}  # More rows for complex analysis
                
                elif isinstance(result, pd.Series):
                    if len(result) == 0:
                        return {"type": "text", "content": f"Multi-table analysis found no results for: {question}"}
                    else:
                        result_df = result.reset_index()
                        return {"type": "dataframe", "content": result_df.head(50)}
                
                elif isinstance(result, (int, float)):
                    return {"type": "text", "content": f"Multi-table analysis result: {result}"}
                
                elif result is None: # Explicitly handle None if no other type matches
                    return {"type": "text", "content": "Analysis completed, no specific data result to display (e.g., plot displayed or operation had no direct return value)."}
                
                else:
                    # Catch-all for other types of results or messages
                    return {"type": "text", "content": f"Multi-table analysis: {str(result)}"}
            
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

def run_lotus_query(tables: List[Tuple[str, pd.DataFrame]], question: str, model: str = "mistral", backend: str = "ollama", api_key: str = None) -> Dict[str, Any]:
    """Run semantic query using enhanced multi-table prompting (no Lotus-AI required)"""
    lotus_llm = LotusLLM(tables, model=model, backend=backend, api_key=api_key)
    return lotus_llm.run_lotus(question)