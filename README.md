## 2023-02 Machine Learning and Finance Time Series Forecasting Final Project

성균관대학교 경제학과 김겨레  
성균관대학교 경제학과 김민호 (석사과정)

----
**연구 개요**  
VKOSPI는 KOSPI200에 대한 옵션가격을 통해 산출되는 시장 변동성 지표이다. VKOSPI는 시장 참여자의 기대를 반영하는 정보이기 때문에 이를 활용하여 시장 변동성의 증감을 예측하여 시장 상황에 선제적으로 대응할 수 있다. 본 연구는 기존 변동성 예측 연구에서 주로 활용하던 Heteroskedasticity Autoregressive (HAR) 모형을 벤치마크 모형으로 두고, 머신러닝 알고리즘의 변동성 예측력을 평가하였다. 본 연구에서는 Long Term Short Memory (LSTM)를 주된 머신러닝 알고리즘으로 활용했다. 해당 알고리즘은 설명변수가 들어갈 수 있는 레이어를 여러 계층으로 나눌 수 있다. 이런 점을 활용하여 성격이 다른 설명변수를 서로 다른 레이어에 넣어 기존의 경제 예측 머신러닝 알고리즘과 차별화된 방법론을 사용하였다. 또한 본 연구에서는 시장의 변동성에 영향을 줄 것으로 보이지만, 기존에는 활용하지 않았던 반대매매, 미수금 등 설명변수를 추가로 도입하여 예측을 시도하였다. 예측 결과를 비교한 결과, 1주, 2주, 4주의 단기 예측에서는 기존의 계량경제학 모형이 머신러닝 알고리즘에 비해 우월한 것으로 나타나지만, 8주에서는 본 연구에서 도입한 LSTM이 우월한 성과를 보여주는 것으로 나타났다. 또한, Boruta 알고리즘을 통해 중요한 변수를 추출한 결과, 미수금, 반대매매 등 본 연구에서 새롭게 도입한 변수가 비중 있게 추출되었다. 비록 단기 예측에서는 본 연구에서 도입한 알고리즘이 기존 모형보다 부족한 성과를 기록했지만, 중장기 예측에서는 해당 알고리즘이 유용하게 사용될 수 있음을 시사한다.

----
**1. 본 연구에서 활용한 설명변수 목록**  
![image](https://github.com/popper6508/202302_MLFTF_Project/assets/118153199/4b81c3ac-8585-465f-940d-413c5948b8a4)

**2. 본 연구에서 활용한 Multi-Layer LSTM 구조**  
![image](https://github.com/popper6508/202302_MLFTF_Project/assets/118153199/8ef13ff7-5b44-4bb4-b7f6-7bfa051c7b78)
