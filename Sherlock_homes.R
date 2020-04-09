# This is based on Julia Silge's brilliant Sherlock Holmes tutorial
# https://github.com/juliasilge/sherlock-holmes

#########################################################################
# Part 1 - Prepataion
#########################################################################

library(tidyverse)
library(tidytext)
library(gutenbergr)

# Download the adventures of Sherlock Holmes
sherlock_raw <- gutenberg_download(1661)

# Use str_detect to remove the book title and contents page
sherlock <- sherlock_raw %>%
  mutate(story = ifelse(str_detect(text, "ADVENTURE"),
                        text,
                        NA)) %>%
  fill(story) %>%
  filter(story != "THE ADVENTURES OF SHERLOCK HOLMES") %>%
  mutate(story = factor(story, levels = unique(story)))


# Create a 'tidy' dataset, where Story, line, and word are columns.
# Each line 
tidy_sherlock <- sherlock %>%
  mutate(line = row_number()) %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  filter(word != "holmes")


# Lets look at the top rows
tidy_sherlock

# What is the count of words
tidy_sherlock %>%
  count(word, sort = TRUE)


# TF-IDF
# Term frequency, inverse document frequency
# View the top 10 TF-IDF scored words per story
tidy_sherlock %>%
  count(story, word, sort = TRUE) %>%
  bind_tf_idf(word, story, n) %>%
  arrange(-tf_idf) %>%
  group_by(story) %>%
  top_n(10) %>%
  ungroup %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(word, tf_idf, fill = story)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ story, scales = "free") +
  coord_flip()



############################################################################################
# Part 2: Topic models
############################################################################################

library(topicmodels)

sherlock_dtm <- tidy_sherlock %>%
  count(story, word, sort = TRUE) %>%
  cast_dtm(story, word, n)


# Build LDA model
topic_model <- LDA(sherlock_dtm, k = 6, method = "Gibbs")


# per-document-per-topic probabilities
td_gamma <- tidy(topic_model, matrix = "gamma",                    
                 document_names = rownames(sherlock_dtm))
td_gamma


# Lets plot them:
ggplot(td_gamma, aes(gamma, fill = as.factor(topic))) +
  geom_histogram(show.legend = FALSE) +
  facet_wrap(~ topic, ncol = 3) +
  labs(title = "Distribution of probability for each topic",
       y = "Number of documents", x = expression(gamma))



#  We can then assign each word in each document to a topic
assignments <- augment(topic_model, sherlock_dtm)
assignments
