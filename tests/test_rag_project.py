import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIRS = {
    "simple_rag": ROOT / "app" / "simple_rag",
    "advanced_rag": ROOT / "app" / "advanced_rag",
}
MODULE_NAMES = [
    "chunk",
    "embedding",
    "elasticsearch_client",
    "llm_client",
    "schemas",
    "main",
    "reranking",
    "query_rewriter",
    "response_evaluation",
]


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts):
        vectors = [[float(len(text)), float(index)] for index, text in enumerate(texts)]
        return FakeEncodedVectors(vectors)


class FakeEncodedVectors:
    def __init__(self, vectors):
        self.vectors = vectors

    def tolist(self):
        return self.vectors


class FakeIndicesClient:
    def __init__(self):
        self.store = {}

    def get(self, index):
        if index not in self.store:
            raise FakeNotFoundError(index)
        return {index: self.store[index]}

    def create(self, index, body):
        self.store[index] = {"body": body}
        return {"acknowledged": True}

    def exists(self, index):
        return index in self.store


class FakeElasticsearch:
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers or {}
        self.indices = FakeIndicesClient()
        self._count = 0
        self._hits = []

    def search(self, index, query, size):
        hits = self._hits[:size]
        return {"hits": {"hits": hits}}

    def count(self, index):
        return {"count": self._count}

    def index(self, index, id, document):
        return {"result": "created", "_id": id, "_index": index, "_source": document}


class FakeNotFoundError(Exception):
    pass


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data


class FakePromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)


class FakeOllamaLLM:
    def __init__(self, model, stop=None):
        self.model = model
        self.stop = stop or []
        self.invoke_response = "[]"
        self.call_response = "rewritten query"

    def invoke(self, prompt):
        return self.invoke_response

    def __call__(self, prompt):
        return self.call_response


class FakeUploadFile:
    def __init__(self, content, filename="test.txt"):
        self._content = content
        self.filename = filename

    async def read(self):
        return self._content


def clear_project_modules():
    for name in MODULE_NAMES:
        sys.modules.pop(name, None)


def install_dependency_stubs():
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = sentence_transformers

    elasticsearch = types.ModuleType("elasticsearch")
    elasticsearch.Elasticsearch = FakeElasticsearch
    sys.modules["elasticsearch"] = elasticsearch

    elasticsearch_exceptions = types.ModuleType("elasticsearch.exceptions")
    elasticsearch_exceptions.NotFoundError = FakeNotFoundError
    sys.modules["elasticsearch.exceptions"] = elasticsearch_exceptions

    requests = types.ModuleType("requests")
    requests.post = lambda *args, **kwargs: FakeResponse(
        200, {"response": "stubbed answer"}, ""
    )
    sys.modules["requests"] = requests

    langchain = types.ModuleType("langchain")
    langchain.PromptTemplate = FakePromptTemplate
    sys.modules["langchain"] = langchain

    langchain_ollama = types.ModuleType("langchain_ollama")
    langchain_ollama.OllamaLLM = FakeOllamaLLM
    sys.modules["langchain_ollama"] = langchain_ollama


def load_module(project, module_name):
    clear_project_modules()
    install_dependency_stubs()
    for candidate_dir in PROJECT_DIRS.values():
        candidate_dir_str = str(candidate_dir)
        while candidate_dir_str in sys.path:
            sys.path.remove(candidate_dir_str)
    project_dir = str(PROJECT_DIRS[project])
    sys.path.insert(0, project_dir)
    return importlib.import_module(module_name)


class ChunkingTests(unittest.TestCase):
    def test_simple_chunking_returns_overlapping_chunks(self):
        module = load_module("simple_rag", "chunk")
        result = module.chunking("abcdefghij", chunk_size=4, overlap=1)
        self.assertEqual(result, ["abcd", "defg", "ghij", "j"])

    def test_simple_chunking_rejects_invalid_overlap(self):
        module = load_module("simple_rag", "chunk")
        with self.assertRaises(ValueError):
            module.chunking("abc", chunk_size=3, overlap=3)

    def test_advanced_chunking_rejects_invalid_overlap(self):
        module = load_module("advanced_rag", "chunk")
        with self.assertRaises(ValueError):
            module.chunking("abc", chunk_size=2, overlap=2)


class EmbeddingTests(unittest.TestCase):
    def test_simple_embed_texts_returns_list_vectors(self):
        module = load_module("simple_rag", "embedding")
        result = module.embed_texts(["a", "abc"])
        self.assertEqual(result, [[1.0, 0.0], [3.0, 1.0]])

    def test_advanced_embed_texts_returns_list_vectors(self):
        module = load_module("advanced_rag", "embedding")
        result = module.embed_texts(["hello"])
        self.assertEqual(result, [[5.0, 0.0]])


class LlmClientTests(unittest.TestCase):
    def test_simple_ask_llm_returns_response_body(self):
        module = load_module("simple_rag", "llm_client")
        with patch.object(module.requests, "post", return_value=FakeResponse(200, {"response": "ok"})):
            self.assertEqual(module.ask_llm("prompt"), "ok")

    def test_simple_ask_llm_raises_on_non_200(self):
        module = load_module("simple_rag", "llm_client")
        with patch.object(module.requests, "post", return_value=FakeResponse(500, text="failure")):
            with self.assertRaises(Exception) as ctx:
                module.ask_llm("prompt")
        self.assertIn("Ollama API error", str(ctx.exception))

    def test_advanced_ask_llm_wraps_transport_errors(self):
        module = load_module("advanced_rag", "llm_client")
        with patch.object(module.requests, "post", side_effect=RuntimeError("boom")):
            with self.assertRaises(Exception) as ctx:
                module.ask_llm("prompt")
        self.assertIn("LLM問い合わせ失敗", str(ctx.exception))


class ElasticsearchClientTests(unittest.TestCase):
    def test_index_exists_true_and_false(self):
        module = load_module("simple_rag", "elasticsearch_client")
        module.es.indices.store["exists"] = {}
        self.assertTrue(module.index_exists("exists"))
        self.assertFalse(module.index_exists("missing"))

    def test_create_index_calls_indices_create_for_missing_index(self):
        module = load_module("simple_rag", "elasticsearch_client")
        with patch.object(module.es.indices, "create", wraps=module.es.indices.create) as create_mock:
            module.create_index("new-index")
        create_mock.assert_called_once()

    def test_add_document_indexes_when_index_exists(self):
        module = load_module("simple_rag", "elasticsearch_client")
        module.es.indices.store["target"] = {}
        with patch.object(module.es, "index", wraps=module.es.index) as index_mock:
            module.add_document("target", "doc-1", "body", [0.1, 0.2])
        index_mock.assert_called_once()

    def test_add_document_skips_when_index_missing(self):
        module = load_module("simple_rag", "elasticsearch_client")
        with patch.object(module.es, "index", wraps=module.es.index) as index_mock:
            module.add_document("missing", "doc-1", "body", [0.1, 0.2])
        index_mock.assert_not_called()

    def test_search_similar_returns_content_and_scores(self):
        module = load_module("simple_rag", "elasticsearch_client")
        module.es._hits = [
            {"_source": {"content": "doc-a"}, "_score": 1.2},
            {"_source": {"content": "doc-b"}, "_score": 1.1},
        ]
        result = module.search_similar([0.1], top_k=2, index_name="idx")
        self.assertEqual(
            result,
            [{"content": "doc-a", "score": 1.2}, {"content": "doc-b", "score": 1.1}],
        )

    def test_search_bm25_propagates_search_errors(self):
        module = load_module("simple_rag", "elasticsearch_client")
        with patch.object(module.es, "search", side_effect=RuntimeError("search failed")):
            with self.assertRaises(RuntimeError):
                module.search_bm25("query", top_k=1, index_name="idx")

    def test_search_keyword_uses_keyword_field_response_shape(self):
        module = load_module("advanced_rag", "elasticsearch_client")
        module.es._hits = [{"_source": {"content": "exact"}, "_score": 4.0}]
        result = module.search_keyword("exact", top_k=1, index_name="idx")
        self.assertEqual(result, [{"content": "exact", "score": 4.0}])

    def test_get_documents_returns_empty_list_on_error(self):
        module = load_module("advanced_rag", "elasticsearch_client")
        with patch.object(module.es, "search", side_effect=RuntimeError("broken")):
            self.assertEqual(module.get_documents("idx"), [])


class SchemaTests(unittest.TestCase):
    def test_simple_query_response_model_accepts_expected_payload(self):
        module = load_module("simple_rag", "schemas")
        payload = module.QueryResponse(
            answer="done",
            docs=[module.DocItem(content="text", score=0.9)],
        )
        self.assertEqual(payload.answer, "done")
        self.assertEqual(payload.docs[0].content, "text")

    def test_advanced_query_request_requires_fields(self):
        module = load_module("advanced_rag", "schemas")
        with self.assertRaises(Exception):
            module.QueryRequest(question="q")


class SimpleMainTests(unittest.TestCase):
    def test_index_docs_reads_files_chunks_and_indexes(self):
        module = load_module("simple_rag", "main")
        uploads = [FakeUploadFile(b"abcdef"), FakeUploadFile(b"ghijkl")]
        with patch.object(module, "create_index") as create_index_mock, patch.object(
            module, "chunking", side_effect=[["ab", "cd"], ["gh"]]
        ), patch.object(module, "embed_texts", return_value=[[1], [2], [3]]), patch.object(
            module, "add_document"
        ) as add_document_mock:
            result = asyncio.run(module.index_docs(index_name="idx", documents=uploads))
        create_index_mock.assert_called_once_with("idx")
        self.assertIn("3 件のチャンク", result["message"])
        self.assertEqual(add_document_mock.call_count, 3)

    def test_index_docs_wraps_decode_errors_in_http_exception(self):
        module = load_module("simple_rag", "main")
        uploads = [FakeUploadFile(b"\xff")]
        with self.assertRaises(Exception) as ctx:
            asyncio.run(module.index_docs(index_name="idx", documents=uploads))
        self.assertIn("UTF-8で読めないファイルがあります", str(ctx.exception))

    def test_query_answer_builds_prompt_from_retrieved_docs(self):
        module = load_module("simple_rag", "main")
        req = module.QueryRequest(question="Q?", top_k=2, index_name="idx")
        docs = [{"content": "A", "score": 1.0}, {"content": "B", "score": 0.8}]
        with patch.object(module, "embed_texts", return_value=[[0.1]]), patch.object(
            module, "search_similar", return_value=docs
        ), patch.object(module, "ask_llm", return_value="answer") as ask_mock:
            result = module.query_answer(req)
        self.assertEqual(result.answer, "answer")
        self.assertEqual([doc.model_dump() for doc in result.docs], docs)
        self.assertIn("A\nB", ask_mock.call_args[0][0])

    def test_get_index_returns_documents(self):
        module = load_module("simple_rag", "main")
        with patch.object(module.es, "count", return_value={"count": 2}), patch.object(
            module.es,
            "search",
            return_value={"hits": {"hits": [{"_source": {"content": "x"}}, {"_source": {"content": "y"}}]}},
        ):
            result = module.get_index("idx")
        self.assertEqual(result, {"index_name": "idx", "documents": ["x", "y"]})

    def test_get_index_wraps_errors_in_http_exception(self):
        module = load_module("simple_rag", "main")
        with patch.object(module.es, "count", side_effect=RuntimeError("nope")):
            with self.assertRaises(Exception) as ctx:
                module.get_index("idx")
        self.assertIn("インデックス取得失敗", str(ctx.exception))


class AdvancedHelpersTests(unittest.TestCase):
    def test_reranking_returns_original_candidates_for_empty_response(self):
        module = load_module("advanced_rag", "reranking")
        candidates = [{"content": "a", "score": 1.0}]
        fake_llm = FakeOllamaLLM("llama3")
        fake_llm.invoke_response = ""
        result = module.reranking("query", candidates, fake_llm)
        self.assertEqual(result, candidates)

    def test_reranking_returns_original_candidates_for_invalid_json(self):
        module = load_module("advanced_rag", "reranking")
        candidates = [{"content": "a", "score": 1.0}]
        fake_llm = FakeOllamaLLM("llama3")
        fake_llm.invoke_response = "not-json"
        result = module.reranking("query", candidates, fake_llm)
        self.assertEqual(result, candidates)

    def test_reranking_reorders_candidates_for_valid_json(self):
        module = load_module("advanced_rag", "reranking")
        candidates = [{"content": "first", "score": 0.1}, {"content": "second", "score": 0.2}]
        fake_llm = FakeOllamaLLM("llama3")
        fake_llm.invoke_response = '[{"index": 2, "score": 0.95}, {"index": 1, "score": 0.5}]'
        result = module.reranking("query", candidates, fake_llm)
        self.assertEqual(result[0]["content"], "second")
        self.assertEqual(result[0]["score"], 0.95)

    def test_rewrite_query_returns_trimmed_text(self):
        module = load_module("advanced_rag", "query_rewriter")
        fake_llm = FakeOllamaLLM("llama3")
        fake_llm.call_response = "  rewritten  "
        with patch.object(module, "llm", fake_llm):
            self.assertEqual(module.rewrite_query("original"), "rewritten")

    def test_evaluate_answer_returns_float_score(self):
        module = load_module("advanced_rag", "response_evaluation")
        with patch.object(module, "ask_llm", return_value="0.75"):
            score = module.evaluate_answer("q", "a", [{"content": "doc"}])
        self.assertEqual(score, 0.75)

    def test_evaluate_answer_returns_zero_for_invalid_score(self):
        module = load_module("advanced_rag", "response_evaluation")
        with patch.object(module, "ask_llm", return_value="not-a-number"):
            score = module.evaluate_answer("q", "a", [{"content": "doc"}])
        self.assertEqual(score, 0.0)


class AdvancedMainTests(unittest.TestCase):
    def test_advanced_query_answer_uses_reranked_context_and_rewritten_query(self):
        module = load_module("advanced_rag", "main")
        req = module.QueryRequest(question="Q?", top_k=2, index_name="idx")
        docs = [{"content": "raw-a", "score": 1.0}, {"content": "raw-b", "score": 0.8}]
        reranked = [{"content": "better", "score": 9.0}]
        with patch.object(module, "embed_texts", return_value=[[0.1]]), patch.object(
            module, "search_similar", return_value=docs
        ), patch.object(module, "reranking", return_value=reranked), patch.object(
            module, "rewrite_query", return_value="rewritten question"
        ), patch.object(module, "ask_llm", return_value="answer") as ask_mock:
            result = module.query_answer(req)
        self.assertEqual(result.answer, "answer")
        self.assertEqual([doc.model_dump() for doc in result.docs], reranked)
        self.assertIn("better", ask_mock.call_args[0][0])
        self.assertIn("rewritten question", ask_mock.call_args[0][0])

    def test_advanced_get_index_wraps_errors(self):
        module = load_module("advanced_rag", "main")
        with patch.object(module.es, "count", side_effect=RuntimeError("nope")):
            with self.assertRaises(Exception) as ctx:
                module.get_index("idx")
        self.assertIn("インデックス取得失敗", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
