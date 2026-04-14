file_path = "/home/tommy/Desktop/projects/sdp_icl/sdp-icl/results/sdp/val_5000/sdp_qa_llama_sanitized_n100_5000_results.jsonl"
target = 5000

with open(file_path, "rb") as f:
    count = sum(1 for _ in f)

progress = (count / target) * 100

print(f"目前資料筆數: {count}")
print(f"完成度: {progress:.2f}%")