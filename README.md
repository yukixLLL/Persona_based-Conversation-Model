# Persona based Conversation Model

*EPFL Semester Project, Fall 2019. By Yueran Liang*

All the detail of this project is in the [report](https://github.com/yukixLLL/Persona_based-Conversation-Model/blob/master/Persona_report.pdf).

## Description of important documents:
**persona_python** contains basic persona-based model. In the folder, `main_SM.py` and `main_SAM.py` corresponds to run the training in Speaker Model and Speaker-Addressee Model respectively. For validation, `main_validation.py` is provided.

**vae_python** contains persona-based CVAE model. In the folder, `vae_SM` and `vae_SAM` corresponds to run the training in Speaker Model and Speaker-Addressee Model respectively. For validation, `main_validation.py` is provided.

**vae_gpu** also contains persona_based CVAE model. But the codes are for running on gpu in a distributed way.

Responses are generated using the codes in `.ipynb` file in corresponding **xxx_python** folder.

## Reference
[1] Alan Ritter, Colin Cherry, and William B Dolan. (2011). Data-driven Response Generation in Social Media. In *Proceedingsof the Conference on Empirical Methods in Natural Language Processing*, pages 583593.

[2] Iulian V Serban, Alessandro Sordoni, Yoshua Bengio, AaronCourville, and Joelle Pineau. (2015). Building End-to-end Di-alogue Systems Using Generative Hierarchical Neural NetworkModels. In *Proc. of AAAI*.

[3] Oriol Vinyals and Quoc Le. (2015). A Neural ConversationalModel. In *Proc. of ICML Deep Learning Workshop*.

[4] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and BillDolan. (2016a). A Diversity-promoting Objective Function forNeural Conversation Models. In *Proc. of NAACL-HLT*.

[5] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and BillDolan. (2016b). A persona-based neural conversation model. *arXiv preprint arXiv:1603.06155*.

[6] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. (2014). Sequence to Sequence Learning With Neural Networks. In *Advances in neural information processing systems (NIPS)*, pages 3104–3112.

[7] Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam,Douwe Kiela, Jason Weston. (2018). Personalizing Dialogue Agents: I have a dog, do you have pets too? *arXiv preprint arXiv:1801.07243*.

[8] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. (2015).End-to-end memory networks. InAdvances in *Neural Information Processing Systems*, pages 2440–2448.

[9] Andrea Madotto, Zhaojiang Lin, Chien-Sheng Wu, PascaleFung (2019). Personalizing Dialogue Agents via Meta-Learning. In *Association for Computational Linguistics*, pages 5454–5459.

[10] Chelsea Finn, Pieter Abbeel, and Sergey Levine. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning-Volume 70*, pages 1126–1135. JMLR. org.

[11] Oluwatobi Olabiyi, Anish Khazane, Alan Salimov, Erik T.Mueller (2019). An Adversarial Learning Framework For A Persona-Based  Multi-Turn  Dialogue  Model *arXiv  preprintarXiv:1905.01992*.

[12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu,David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Ben-gio. (2014). Generative Adversarial Networks In *Proceedings of the International Conference on Neural Information Processing Systems (NIPS 2014)*, pages 2672–2680.

[13] Sepp Hochreiter and J ̈urgen Schmidhuber. (1997). Long Short-term Memory. *Neural computation*, 9(8):1735–1780.

[14] Dzmitry Bahdanau, KyungHyun Cho and Yoshua Bengio. (May19, 2016). Neural Machine Translation by Jointly Learning toAlign and Translate. *arXiv preprint arXiv:1409.0473*.

[15] Diederik P Kingma and Max Welling. (May 1, 2014). Auto-encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.

[16] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wier-stra. (May 30, 2014). Stochastic Backpropagation and Approx-imate Inference in Deep Generative Models. *arXiv preprintarXiv:1401.4082*.

[17] Xinchen Yan, Jimei Yang, Kihyuk Sohn, and Honglak Lee.(2015). Attribute2image: Conditional image generation from vi-sual attributes. *arXiv preprint arXiv:1512.00570*.

[18] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. (2015). Learningstructured output representation using deep conditional gen-erative models. In *Advances in Neural Information Processing Systems*, pages 3483–3491.

[19] Tiancheng  Zhao,  Ran  Zhao,  Maxine  Eskenazi  (2017). Learning Discourse-level Diversity for Neural Dialog Modelsusing  Conditional  Variational  Autoencoders *arXiv preprint arXiv:1703.10960*.

[20] Kihyuk Sohn, Honglak Lee, and Xinchen Yan. (2015). Learningstructured output representation using deep conditional gen-erative models. In *Advances in Neural Information ProcessingSystems*, pages 3483–3491.

[21] Samuel R Bowman, Luke Vilnis, Oriol Vinyals, Andrew M Dai,Rafal Jozefowicz, and Samy Bengio. (2015). Generating sentencesfrom a continuous space. *arXiv preprint arXiv:1511.06349*.

[22] Mike Schuster and Kuldip K Paliwal. (1997). BidirectionalRecurrent Neural Networks. *IEEE Transactions on Signal Pro-cessing*, 45(11):2673–2681.

[23] J ̈org Tiedemann. (2009). News from OPUS – a collection ofmultilingual parallel corpora with tools and interfaces. In *Re-cent advances in natural language processing*, volume 5, pages237–248.

[24] Franz Josef Och. (2003). Minimum Error Rate Training in Sta-tistical Machine Translation. In *Proceedings of the 41st AnnualMeeting of the Association for Computational Linguistics*, pages160–167, Sapporo, Japan, July. Association for ComputationalLinguistics.
