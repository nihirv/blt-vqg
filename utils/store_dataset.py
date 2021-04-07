"""Transform all the IQ VQA dataset into a hdf5 dataset.
"""

import base64
import csv
import pickle
import sys
from PIL import Image
from numpy.lib.type_check import imag
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar
import copy
from tqdm import tqdm

from utils.train_utils import Vocabulary
from utils.vocab import build_vocab, load_vocab
from utils.vocab import process_text

csv.field_size_limit(sys.maxsize)


def create_answer_mapping(annotations, ans2cat):
    """Returns mapping from question_id to answer.

    Only returns those mappings that map to one of the answers in ans2cat.

    Args:
        annotations: VQA annotations file.
        ans2cat: Map from answers to answer categories that we care about.

    Returns:
        answers: Mapping from question ids to answers.
        image_ids: Set of image ids.
    """
    answers = {}
    image_ids = set()
    for q in annotations['annotations']:
        question_id = q['question_id']
        answer = q['multiple_choice_answer']
        if answer in ans2cat:
            answers[question_id] = answer
            image_ids.add(q['image_id'])
    return answers, image_ids


def read_image_features_tsv(tsv_in_file, FIELDNAMES=['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']):
    image_feature_data = {}
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
    for i, item in tqdm(enumerate(reader)):
        item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])
        item['num_boxes'] = int(item['num_boxes'])
        for field in ['boxes', 'features']:
            try:
                item[field] = np.frombuffer(
                    base64.b64decode(item[field].encode() + b'==='),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
            except:
                print("Error processing file {}".format(item["image_id"]))
                continue

        boxes = copy.deepcopy(item["boxes"])
        boxes[:, [0, 2]] /= item['image_w']
        boxes[:, [1, 3]] /= item['image_h']

    # Normalized box areas
        areas = (boxes[:, 2] - boxes[:, 0]) * \
            (boxes[:, 3] - boxes[:, 1])

        image_feature_data[item['image_id']] = {
            "normalized_boxes_area": np.c_[boxes, areas],
            "features": item["features"]
        }
    return image_feature_data


def save_dataset(image_dir, questions, annotations, vocab, ans2cat, output,
                 im_size=224, max_q_length=20, max_a_length=4,
                 with_answers=False, train_or_val="train", num_objects=36):
    """Saves the Visual Genome images and the questions in a hdf5 file.

    Args:
        image_dir: Directory with all the images.
        questions: Location of the questions.
        annotations: Location of all the annotations.
        vocab: Location of the vocab file.
        ans2cat: Mapping from answers to category.
        output: Location of the hdf5 file to save to.
        im_size: Size of image.
        max_q_length: Maximum length of the questions.
        max_a_length: Maximum length of the answers.
        with_answers: Whether to also save the answers.
    """
    # Load the data.
    with open(annotations) as f:
        annos = json.load(f)
    with open(questions) as f:
        questions = json.load(f)

    # Load RCNN features
    print("Loading RCNN features. This may take a while")
    image_feature_data = {}

    filepaths = [
        "data/image_features/updown/trainval/karpathy_test_resnet101_faster_rcnn_genome.tsv",
        "data/image_features/updown/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.0",
        "data/image_features/updown/trainval/karpathy_train_resnet101_faster_rcnn_genome.tsv.1",
        'data/image_features/updown/trainval/karpathy_val_resnet101_faster_rcnn_genome.tsv',
        'data/image_features/updown/test2014/test2014_resnet101_faster_rcnn_genome.tsv.0',
        'data/image_features/updown/test2014/test2014_resnet101_faster_rcnn_genome.tsv.1',
        'data/image_features/updown/test2014/test2014_resnet101_faster_rcnn_genome.tsv.2',
        'data/image_features/updown/test2015/test2015_resnet101_faster_rcnn_genome.tsv'
    ]
    for filepath in filepaths:
        print("Loading file...")
        with open(filepath, "r") as tsv_in_file:
            ifd = read_image_features_tsv(tsv_in_file)
            image_feature_data.update(ifd)
    print("Loaded RCNN features!")

    # Get the mappings from qid to answers.
    qid2ans, image_ids = create_answer_mapping(annos, ans2cat)
    total_questions = len(list(qid2ans.keys()))
    total_images = len(image_ids)
    print("Number of images to be written: %d" % total_images)
    print("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions, max_a_length), dtype='i')
    d_answer_types = h5file.create_dataset(
        "answer_types", (total_questions,), dtype='i')
    d_image_ids = h5file.create_dataset(
        "image_ids", (total_questions,), dtype='i')
    d_rcnn_features = h5file.create_dataset(
        "rcnn_features", (total_questions, num_objects, 2048), dtype='f')
    d_rcnn_locations = h5file.create_dataset(
        "rcnn_locations", (total_questions, num_objects, 5), dtype='f')

    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total_questions)
    bar.start()
    i_index = 0
    q_index = 0
    done_img2idx = {}
    found_images = set()
    not_found_images = set()
    for entry in questions['questions']:
        image_id = entry['image_id']
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:
            try:
                path = "COCO_%s2014_%d.jpg" % (train_or_val, image_id)
                image = Image.open(os.path.join(
                    image_dir, path)).convert('RGB')
            except IOError:
                try:
                    path = "COCO_%s2014_%012d.jpg" % (train_or_val, image_id)
                    image = Image.open(os.path.join(
                        image_dir, path)).convert('RGB')
                except:
                    print("COULD NOT FIND IMAGE {}".format(path))
                    continue
            image = transform(image)
            d_images[i_index, :, :, :] = np.array(image)
            done_img2idx[image_id] = i_index
            i_index += 1
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)
        d_questions[q_index, :length] = q
        answer = qid2ans[question_id]
        a, length = process_text(answer, vocab,
                                 max_length=max_a_length)
        d_answers[q_index, :length] = a
        d_answer_types[q_index] = int(ans2cat[answer])
        d_indices[q_index] = done_img2idx[image_id]
        d_image_ids[q_index] = image_id

        rcnn_features_zeros = np.zeros((num_objects, 2048), dtype=np.float32)
        rcnn_normalised_boxes = np.zeros((num_objects, 5), dtype=np.float32)

        try:
            relevant_image_feature_object = image_feature_data[image_id]
            found_images.add(image_id)
        except:
            # print("Skipping file {} due to an error. Most like the file could not be found.".format(
            #     image_id))
            not_found_images.add(image_id)
            continue

        image_features, normalised_boxes = relevant_image_feature_object[
            "features"], relevant_image_feature_object["normalized_boxes_area"]
        len_features = image_features.shape[0]
        if len_features > num_objects:
            image_features = image_features[:num_objects]
            normalised_boxes = normalised_boxes[:num_objects]
            rcnn_features_zeros = image_features
            rcnn_normalised_boxes = normalised_boxes
        else:
            rcnn_features_zeros[:len_features] = image_features
            rcnn_normalised_boxes[:len_features] = normalised_boxes

        d_rcnn_features[q_index] = rcnn_features_zeros
        d_rcnn_locations[q_index] = rcnn_normalised_boxes

        q_index += 1
        bar.update(q_index)
    h5file.close()
    print("Number of images written: %d" % i_index)
    print("Number of QAs written: %d" % q_index)
    print("Number of images found ({}) vs not found ({})".format(
        len(found_images), len(not_found_images)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='data/vqa/train2014',
                        help='directory for resized images')
    parser.add_argument('--questions', type=str,
                        default='data/vqa/v2_OpenEnded_mscoco_'
                        'train2014_questions.json',
                        help='Path for train annotation file.')
    parser.add_argument('--annotations', type=str,
                        default='data/vqa/v2_mscoco_'
                        'train2014_annotations.json',
                        help='Path for train annotation file.')
    parser.add_argument('--cat2ans', type=str,
                        default='data/vqa/iq_dataset.json',
                        help='Path for the answer types.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output', type=str,
                        default='data/processed/iq_dataset.hdf5',
                        help='directory for resized images.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=4,
                        help='maximum sequence length for answers.')

    # Train or Val?
    parser.add_argument('--val', type=bool, default=False,
                        help="whether we're working iwth the validation set or not")
    args = parser.parse_args()

    ans2cat = {}
    with open(args.cat2ans) as f:
        cat2ans = json.load(f)
    cats = sorted(cat2ans.keys())
    with open(args.cat2name, 'w') as f:
        json.dump(cats, f)
    for cat in cat2ans:
        for ans in cat2ans[cat]:
            ans2cat[ans] = cats.index(cat)

    train_or_val = "train"
    if args.val == True:
        train_or_val = "val"
        vocab = pickle.load(open("vocab.pkl", "rb"))
        # vocab.save(args.vocab_path)
        # vocab = load_vocab(args.vocab_path)
    else:
        # vocab = build_vocab('data/vqa/v2_OpenEnded_mscoco_train2014_questions.json', 'data/vqa/iq_dataset.json', 4)
        vocab = pickle.load(open("vocab.pkl", "rb"))
        vocab.save(args.vocab_path)

    save_dataset(args.image_dir, args.questions, args.annotations, vocab,
                 ans2cat, args.output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, train_or_val=train_or_val)
    print(('Wrote dataset to %s' % args.output))
    # Hack to avoid import errors.
    Vocabulary()
