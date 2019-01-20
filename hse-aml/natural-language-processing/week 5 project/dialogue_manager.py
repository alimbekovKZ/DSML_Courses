import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from utils import *
from sklearn.metrics.pairwise import cosine_similarity

from model import ChatBotModel
import config
import data


def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")

def _find_right_bucket(length):
    """ Find the proper bucket for an encoder input based on its length """
    return min([b for b in range(len(config.BUCKETS))
                if config.BUCKETS[b][0] >= length])


def _construct_response(output_logits, inv_dec_vocab):
    """ Construct a response to the user's encoder input.
    @output_logits: the outputs from sequence to sequence wrapper.
    output_logits is decoder_size np array, each of dim 1 x DEC_VOCAB

    This is a greedy decoder - outputs are just argmaxes of output_logits.
    """
    print(output_logits[0])
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if config.EOS_ID in outputs:
        outputs = outputs[:outputs.index(config.EOS_ID)]
    # Print out sentence corresponding to outputs.
    return " ".join([tf.compat.as_str(inv_dec_vocab[output]) for output in outputs])


def run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
    """ Run one step in training.
    @forward_only: boolean value to decide whether a backward path should be created
    forward_only is set to True when you just want to evaluate on the test set,
    or when you want to the bot to be in chat mode. """
    encoder_size, decoder_size = config.BUCKETS[bucket_id]
    _assert_lengths(encoder_size, decoder_size, encoder_inputs, decoder_inputs, decoder_masks)

    # input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for step in range(encoder_size):
        input_feed[model.encoder_inputs[step].name] = encoder_inputs[step]
    for step in range(decoder_size):
        input_feed[model.decoder_inputs[step].name] = decoder_inputs[step]
        input_feed[model.decoder_masks[step].name] = decoder_masks[step]

    last_target = model.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    # output feed: depends on whether we do a backward step or not.
    if not forward_only:
        output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                       model.gradient_norms[bucket_id],  # gradient norm.
                       model.losses[bucket_id]]  # loss for this batch.
    else:
        output_feed = [model.losses[bucket_id]]  # loss for this batch.
        for step in range(decoder_size):  # output logits.
            output_feed.append(model.outputs[bucket_id][step])

    outputs = sess.run(output_feed, input_feed)
    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.




class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings('/home/aashish/Documents/sentiment_practice1/glove.6B/glove.6B.300d.txt')
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        #question_vec = #### YOUR CODE HERE ####
        #best_thread = #### YOUR CODE HERE ####
        question_vec = question_to_vec(question=question, embeddings=self.word_embeddings, dim=self.embeddings_dim)
        vecq = np.array([question_vec])
        vecc = thread_embeddings
        scores = list(cosine_similarity(vecq, vecc)[0])
        tl = [(i, thread_embeddings[i], scores[i]) for i in range(thread_embeddings.shape[0])]
        stl = sorted(tl, key=lambda x:x[2], reverse=True)
        best_thread = stl[0][0]

        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param

        ########################
        #### YOUR CODE HERE ####
        ########################
        _, self.enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
        self.inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))


        self.model = ChatBotModel(True, batch_size=1)
        self.model.build_graph()
        self.sess = tf.Session()
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        _check_restore_parameters(self.sess, saver)




    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        #prepared_question = #### YOUR CODE HERE ####
        #features = #### YOUR CODE HERE ####
        #intent = #### YOUR CODE HERE ####
        prepared_question = text_prepare(question)
        #### YOUR CODE HERE ####
        features = self.tfidf_vectorizer.transform([prepared_question])
        #### YOUR CODE HERE ####
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.
            max_length = config.BUCKETS[-1][0]
            line = question
            if len(line) > 0 and line[-1] == '\n':
                line = line[:-1]

            token_ids = data.sentence2id(self.enc_vocab, str(line))
            if (len(token_ids) > max_length):
                response = "Max length I can handle is:" + str(max_length)

            bucket_id = _find_right_bucket(len(token_ids))

            encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])],
                                                                            bucket_id,
                                                                            batch_size=1)

            _, _, output_logits = run_step(self.sess, self.model, encoder_inputs, decoder_inputs,
                                           decoder_masks, bucket_id, True)
            response = _construct_response(output_logits, self.inv_dec_vocab)

            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]

            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)

        return self.ANSWER_TEMPLATE % (tag, thread_id)
