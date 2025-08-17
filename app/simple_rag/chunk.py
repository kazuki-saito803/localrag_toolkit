def chunking(text: str, chunk_size: int = 50, overlap: int = 10):
    """
    文字数ベースでテキストをチャンク分割し、重複部分も含める
    text: 分割したいテキスト
    chunk_size: 1チャンクの文字数
    overlap: 前のチャンクと重複させる文字数
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap  # オーバーラップ分を戻す

    return chunks


if __name__ == '__main__':
    input_text = "これはテスト用の長い文章です。" * 50
    chunked_text = chunking(input_text)
    result = {i: j for i, j in enumerate(chunked_text)}
    print(result)