from together import Together

client = Together()

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[
      {
        "role": "user",
        "content": "What are some fun things to do in New York?"
      }
    ]
)
print(response.choices[0].message.content)