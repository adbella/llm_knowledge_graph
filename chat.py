import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pyvis.network import Network
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

SystemPrompt = """You are a helpful assistant for extracting human language knowledge intotriple structures.

## Task
- Extract ALL possible knowledge triples from the given Korean text.
- A triple consists of (subject, predicate, object).
- The subject is the entity being described.
- The predicate describes the action, state, or nature of the subject.
- The object is the target of that predicate.
- Extract explicit and implicit knowledge.
- Entities may represent people, objects, concepts, events, roles, etc.

## Output Requirements

1. Every triple MUST be formatted exactly like this:
(주어, 서술어, 목적어)
2. Each triple MUST be separated by the delimiter:
<|>
3. The entire answer MUST be a single string without a list, without quotes.
4. The answer MUST be in Korean.
5. Include as many knowledge triples as possible.
6. Natural language expressions are allowed in predicate and object.

## Example 1
Input:
"생성 모델은 데이터를 학습하고 새로운 샘플을 생성한다."

Output:
(생성 모델, 데이터, 학습)<|>(생성 모델, 새로운 샘플, 생성)

## Example 2
Input:
"머신러닝 기법 중 선형 회귀(Linear Regression)는 대표적인 회귀 문제에 속하고, 로지스틱 회귀(Logistic Regression)은 대표적인 분류 문제에 속한다.
분류는 이진 분류(Binary Classification)과 다중 클래스 분류(Multi-Class Classification) 등으로 나뉜다."

Output:
(머신러닝, 기법, 선형 회귀)<|>(선형 회귀, 회귀 문제, 속함)<|>(머신러닝, 기법, 로지스틱 회귀)<|>(로지스틱 회귀, 분류 문제, 속함)<|>(분류, 이진 분류, 나뉨)<|>(분류, 다중 클래스 분류, 나뉨)

Now extract knowledge triples from the following context:
{context} """

def parse_output(llm_output):
    if not llm_output:
        return []
    return llm_output.split("<|>")

def graphcategory(parsing_output):
    triples = [i.replace("(","").replace(")","") for i in parsing_output]
    data = [j.split(",") for j in triples]
    categorized_data = [[item[0].strip(), item[2].strip(), item[1].strip()] for item in data]
    return categorized_data

def auto_knowledge_graph(context):
    prompt = PromptTemplate.from_template(SystemPrompt)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )
    chain = prompt | llm
    qa = chain.invoke(context)

    llm_output = qa.content

    parsing_output = parse_output(llm_output)

    categorized_data = graphcategory(parsing_output)
    return categorized_data

context = """지도 학습이란 레이블(Label) 이라는 정답과 함께 학습하는 것을 말한다. 자연어 처리는 대부분 지도 학습에 속한다. 레이블이라는 말 이외에도y, 실제값 등으로 부르기도 한다. 
간단히 말해 선생님이 문제를 내고 그 다음 바로 정답까지 같이 알려주는 방식의 학습 방법이다. 
여러 문제와 답을 같이 학습함으로 미지의 문제에 대한 올바른 답을 예측하고자 하는 방법이다. 
지도학습을 위한 데이터로는 문제와 함께 그 정답까지 같이 알고 잇는 데이터가 선택된다."""

auto_knowledge_graph(context)