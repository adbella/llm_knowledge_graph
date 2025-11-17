import streamlit as st
import json
import requests
import os
from streamlit_agraph import agraph, Node, Edge, Config, TripleStore
from chat import auto_knowledge_graph

st.title("😊 knowledge Graph Generate")

st.markdown("### 문장의 구조도 설명")

user_text = st.text_area(label="문장을 적으세요 (10 ~ 5000자 제한)",
value = """지도 학습이란 레이블(Label) 이라는 정답과 함께 학습하는 것을 말한다. 자연어 처리는 대부분 지도 학습에 속한다. 레이블이라는 말 이외에도y, 실제값 등으로 부르기도 한다. 
간단히 말해 선생님이 문제를 내고 그 다음 바로 정답까지 같이 알려주는 방식의 학습 방법이다. 
여러 문제와 답을 같이 학습함으로 미지의 문제에 대한 올바른 답을 예측하고자 하는 방법이다. 
지도학습을 위한 데이터로는 문제와 함께 그 정답까지 같이 알고 잇는 데이터가 선택된다.""", height=200)

res = None
with st.form('summarize_from', clear_on_submit=True):
    submitted = st.form_submit_button('구조도 생성하기')
    if submitted:
        with st.spinner('구조도를 생성하는 중 입니다...'):
            result = auto_knowledge_graph(user_text)
            res = result

if res:
    nodes = []
    edges = []
    duplication = []

    for item in result:
        if item[0] in duplication:
            pass
        else:
            duplication.append(item[0])
            nodes.append(Node(id=item[0], label=item[0], size=10))

        if item[1] in duplication:
            pass
        else:
            duplication.append(item[1])
            nodes.append(Node(id=item[1], label=item[1], size=10))

        if item[2] in duplication:
            pass
        else:
            duplication.append(item[2])
            nodes.append(Node(id=item[2], label=item[2], size=10))
        
        edges.append(Edge(source=item[0], target=item[1], type="CURVE_SMOOTH"))
        edges.append(Edge(source=item[1], target=item[2], type="CURVE_SMOOTH"))

    config = Config(
            width=750,
            height=950,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=True,
            node={'labelProperty': 'label'},
    )
    return_values = agraph(nodes=nodes, edges=edges, config=config)