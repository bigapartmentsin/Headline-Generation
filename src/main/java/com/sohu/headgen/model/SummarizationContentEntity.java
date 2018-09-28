package com.sohu.headgen.model;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import lombok.Data;

@Data
public class SummarizationContentEntity {

    //所有已分词的句子
    private List<List<String>> sentencesWords;
    //所有完整的句子
    private List<String> sentences;
    //所有词的集合
    private Set<String> wordsSet;
    //句子数
    private int sentencesCount;

    public SummarizationContentEntity() {
        this.sentences = new ArrayList<>();
        this.sentencesWords = new ArrayList<>();
        this.wordsSet = new HashSet<>();
        this.sentencesCount = 0;
    }

    public void addSentence  (String sentence, List<String> sentenceWords) {
        this.sentences.add(sentence);
        this.sentencesWords.add(sentenceWords);
        this.sentencesCount++;
        this.wordsSet.addAll(sentenceWords);
    }

    public void addWord (String word) {
        this.wordsSet.add(word);
    }

    public List<String> getSentenceWords (int i) {
        return this.sentencesWords.get(i);
    }

    public String getSentence (int i) {
        return this.sentences.get(i);
    }

}
