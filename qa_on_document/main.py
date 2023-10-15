# %%
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field


from unicodedata import normalize
import re

from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain

from langchain.chains import (
    create_qa_with_sources_chain,
    RetrievalQA,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

from tiktoken.core import Encoding
import tiktoken

from typing import List
from copy import deepcopy
from textwrap import dedent


import secret


class AnswerWithSources(BaseModel):
    answer: str = Field(description="質問の回答")
    sources: List[int] = Field(description="回答のソースとなるページ番号", nullable=True)


class Utils:
    MODEL_NAME = "gpt-3.5-turbo"
    TOKEN_LIMIT = 4096

    def __init__(self):
        self.model: BaseLanguageModel = self.setup_model()
        self.summary_memo: List[Document] = []

    @staticmethod
    def setup_model(
        language_model: BaseLanguageModel = ChatOpenAI,
    ) -> BaseLanguageModel:
        llm = language_model(temperature=0.0, model="gpt-3.5-turbo")
        return llm

    @staticmethod
    def load_pdf(path) -> List[Document]:
        loader = PyPDFLoader(path)
        pages = loader.load_and_split()

        def remove_nonessential_spaces(s: str) -> str:
            # スペース前後がスペースを必要とする言語の文字種である場合にマッチするパターン
            # 考慮する文字: アルファベット、ギリシャ文字、キリル文字、ヘブライ文字、アラビア文字
            pattern = r"(?<![a-zA-Z0-9\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF]) (?![a-zA-Z0-9\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF])"
            return re.sub(pattern, "", s)

        # テキスト前処理
        for page in pages:
            tmp = normalize("NFKC", page.page_content)
            page.page_content = remove_nonessential_spaces(tmp)
        return pages

    def map_summarization(self, pages: List[Document]):
        map_prompt = PromptTemplate(
            template=dedent(
                """
        あなたの仕事は与えられた文章から重要なエッセンスを抽出することです。
        文章のエッセンスを箇条書きにして簡潔な日本語で書き出してください。
        具体性のない部分は箇条書きから省いても構いません。

        文章:{text}
        """
            ),
            input_variables=["text"],
        )
        combile_prompt = PromptTemplate(
            template=dedent(
                """
        あなたの仕事は箇条書きのリストを整理することです。
        箇条書きのリストが与えられた場合、内容が重複する箇所はひとつの文章にまとめた箇条書きのリストを書き出してください。

        箇条書きのリスト:{text}
        """
            ),
            input_variables=["text"],
        )
        summerize_chain = load_summarize_chain(
            llm=self.model,
            chain_type="map_reduce",
            return_intermediate_steps=True,
            # map_prompt=map_prompt,
            # combine_prompt=combile_prompt,
            # verbose=True,
        )
        summarized: dict = summerize_chain.invoke(pages)

        _pages = deepcopy(pages)
        assert len(_pages) == len(summarized["intermediate_steps"]), "要約結果の数が一致しません"
        for i, e in enumerate(_pages):
            e.page_content = summarized["intermediate_steps"][i]
        return _pages

    def generate_question(
        self, document: List[Document], scene: str, assumed_questioner: str
    ) -> str:
        # TODO: 例文を作成
        examples = [{"question": "", "answer": ""}]

        few_shot_prompt = dedent(
            """
        {% if doc_str %}
        次の文章と例を読んで、{{scene}}の場面で{{assumed_questioner}}の立場として、日本語で質問をひとつ作成してください。
        ---
        文章:
        {{ doc_str }}
        ----
        {% else %}
        {{scene}}の場面で{{assumed_questioner}}の立場として、日本語で質問をひとつ作成してください。
        {% endif %}

        例:
        {% for e in examples %}
        質問文:{{ e.question }}
        {% endfor %}

        質問文:
        """
        )
        prompt = PromptTemplate(
            template=few_shot_prompt,
            input_variables=["examples", "doc_str", "scene", "assumed_questioner"],
            template_format="jinja2",
        )

        # プロンプトが長い場合は、書類本文を要約して短縮する（3回まで）
        join_doc_content = lambda document: "\n".join(
            [d.page_content for d in document]
        )
        create_prompt = lambda document: prompt.format(
            doc_str=join_doc_content(document),
            examples=examples,
            scene=scene,
            assumed_questioner=assumed_questioner,
        )
        prompt_string = create_prompt(document)
        for i in range(3):
            if self._over_token_limit(prompt_string):
                if i == 0:
                    summarized = self.map_summarization(document)
                else:
                    summarized = self.map_summarization(summarized)
                prompt_string = create_prompt(summarized)
                self.summary_memo = summarized
            else:
                break
        # assert not self._over_token_limit(
        #     prompt_string
        # ), f"prompt is over and above api token limit.\nToken:{self._get_token_count(prompt_string)}\nPrompt:{prompt_string}"

        # 質問を作成
        generated_question = self.model.invoke(prompt_string)
        return generated_question

    def generate_answer_with_source(
        self, question: str, document: List[Document]
    ) -> AnswerWithSources:
        """質問に対する回答を生成する。

        Args:
            question (str): 質問文
            document (List[Document]): 質問文の回答元となる文書

        Returns:
            AnswerWithSources: 回答と回答のソースとなるページ番号のリストを含む型
        """
        # ベクトルDBへの保存
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vec_db = Chroma.from_documents(document, embedding_function)

        # 情報ソース（PDFのページ数）付きで回答を生成するQAChainを作成
        qa_chain = create_qa_with_sources_chain(self.model, verbose=False)
        document_prompt = PromptTemplate(
            template="Content: {page_content}\nSource: {page}",
            input_variables=["page_content", "page"],
        )
        final_qa_chain = StuffDocumentsChain(
            llm_chain=qa_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
        )

        retrieval_qa = RetrievalQA(
            retriever=vec_db.as_retriever(), combine_documents_chain=final_qa_chain
        )
        ans_string: str = retrieval_qa.run(question)
        ans_structured: AnswerWithSources = self.parse_answer_with_sources(ans_string)
        return ans_structured

    @staticmethod
    def parse_answer_with_sources(answer: str) -> AnswerWithSources:
        # 応答を構文解析
        # ToDo: 例外時にRetryParserを使う
        print(answer)
        parser = PydanticOutputParser(pydantic_object=AnswerWithSources)
        result = parser.parse(answer)
        return result

    def _get_token_count(self, text: str):
        """テキストのトークン数を見積もる
        Args:
            text (str): テキスト
        Returns:
            int: トークン数
        """
        encoding: Encoding = tiktoken.encoding_for_model(self.MODEL_NAME)
        tokens = encoding.encode(text)
        tokens_count = len(tokens)
        return tokens_count

    def _over_token_limit(self, text: str) -> bool:
        """トークン数が上限を超えているかどうかを判定する
        APIのトークン数上限については次のURLを参照
        Args:
            text (str): プロンプトとなるテキスト
        Returns:
            bool: トークン数が上限を超えている場合はTrue
        """
        return self._get_token_count(text) > self.TOKEN_LIMIT


# %%
util = Utils()
pages = util.load_pdf("../data/pdf/2023.05.25人事異動に関するお知らせ.pdf")
# %%
pages
# %%
question = util.generate_question(pages, "株主総会", "株主")
# %%
print(type(question))
print(question)
# %%
util.summary_memo
# %%
if util.summary_memo:
    answer = util.generate_answer_with_source(str(question), util.summary_memo)
else:
    print("full document")
    answer = util.generate_answer_with_source(str(question), pages)

answer

# %%
# テスト用のFakeLLM
# from langchain.llms.fake import FakeListLLM
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain

# from langchain.callbacks import get_openai_callback


# llm_fake = FakeListLLM(responses=["何かのテキストの要約です。", "違うテキストの要約です。"])
# prompt = PromptTemplate(input_variables=["text"], template="text: {text} Summary:")
# chain = LLMChain(llm=llm_fake, prompt=prompt)
# result = chain.run("テスト１")
# print(result)
# # result = "何かのテキストの要約です。"
# result = chain.run("テスト２")
# print(result)
# # result = "違うテキストの要約です。"
# result = chain.run("テスト３")
# print(result)
# result = chain.run("テスト４")
# print(result)

# result = chain.run("テスト４")
# print(result)
# result = chain.run("テスト４")
# print(result)
