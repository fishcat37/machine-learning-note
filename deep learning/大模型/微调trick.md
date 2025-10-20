## LoRA

在大模型中插入4个低秩矩阵，通过他们的内积就能变成两个和q和v大小一致的矩阵，将他们加到q和k就能改善模型的注意力从而适应特别任务

## QLoRA

将模型先经过量化再做LoRa

## MultipleChoiceLogitsProcessor

在`logits_processor_zoo.vllm`中的`MultipleChoiceLogitsProcessor`能够限制模型输出，比如让他只输出yes或者no，它是通过屏蔽其他token的概率实现的

## vllm

这个库能够大幅提高让模型推理速率

## multiprocessing

这个库能够让模型并行推理，比哦如占用两个进程进行两个同时推理，但只适用于多卡并行