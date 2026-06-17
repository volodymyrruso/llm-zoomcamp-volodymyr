from rag_helper import RAGBase

class ragCourse(RAGBase):
    
    def search(self, query, num_results=5):
        
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(f"Filename: {doc['filename']}")
            lines.append(doc['content'])
            lines.append("")
        return "\n".join(lines).strip()

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )
        return response 

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        
        response = self.llm(prompt)
        
        answer = response.output_text
        input_tokens = response.usage.input_tokens
        
        return answer, input_tokens