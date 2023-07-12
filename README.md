# SplitNN_Experiments

Initial experiments conducted to observe the privacy-utility tradeoff of Split Learning using handwritten digits from the FEMNIST Dataset

For a basic introduction to Split Learning or Distributed Learning in general, please see [Split learning for health: Distributed deep learning without sharing raw patient data](https://arxiv.org/pdf/1812.00564.pdf)

In these experiments we simulate a malicious server conducting a model inversion attack on a client in a single client setting 
for simplicity. In these experiments we modify only the attacker(server) capabilities, and will leave the client defense modifications
for another experiment. We work under the absolute best case scenario assumptions for the attacker, to simulate a worst case scenario
situation for the client. 
These assumptions are:
1) The server has knowledge of the architecture of the client
2) The server's independent dataset is of similar distribution to that of the client

This is an introduction to a series of experiments that will work up to multi-client scenarios using sensitive patient data.

Results will be added soon!

