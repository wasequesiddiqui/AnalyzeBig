import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

text = "Hello this is Waseque Siddiqui here, how are you doing today? I hope you're having a great day!"
tokens = enc.encode(text)
print("Tokens", tokens)