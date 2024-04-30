import copy
import os
import re
import torch
import argparse
from transformers import StoppingCriteria, StoppingCriteriaList
from math import ceil
from PIL import Image
import numpy as np
import torch.backends.cudnn as cudnn
from video_llama.common.logger import setup_logger
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, \
    conv_llava_llama_2
import decord

decord.bridge.set_bridge('torch')
import logging
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms
import pdb
import json
from pathlib import Path
import time
import datetime
from tqdm import tqdm
import random
import pandas as pd

random.seed(1234)

df = pd.read_csv('/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/error_yt_ids.csv', names=['yt_id'])
YT_INVALID_IDS = df['yt_id'].values

COD_DOWNLOADED_VIDEOS = os.listdir('/home/mila/s/subhrajyoti.dasgupta/scratch/videollama/data/cod/videos')

def read_txt(path):
    with open(path, "r") as fin:
        data = fin.readline().strip()
    return data


def load_data(args, anno_path, split=None):
    with open(args.gt_file, 'r') as f:
        data = json.load(f)
    random.shuffle(data)
    return data[:args.num_samples]    # CHANGE LATER ON.


def save_result(args, output_dir, results, split_name='test', format=False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_name = f'{args.task}_{args.dataset}_{split_name}_f{args.num_frames}_result_{args.num_samples}.json'
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f)
    print(file_name, 'saved!')

def generate(chat, gr_videos, user_messages, num_beams, temperature, top_p, n_frms, chat_states=None, img_lists=None, audio_flag=True):
    N = len(user_messages)
    if chat_states is None:
        chat_states = []
        for i in range(N):
            if args.model_type == 'vicuna':
                chat_state = default_conversation.copy()
            else:
                chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            chat_states.append(chat_state)
    if img_lists is None:
        img_lists = [[] for i in range(N)]
        try:
            if audio_flag:
                llm_message = chat.upload_video(gr_videos[0]+'.mp4', chat_states[0], img_lists[0])
            else:
                llm_message = chat.upload_video_without_audio(gr_videos, chat_states, img_lists)
        except:
            print(gr_videos[0]+'.mp4 maybe corrupted!')

    for user_message, chat_state in zip(user_messages, chat_states):
        chat.ask(user_message, chat_state)

    response = chat.answer(conv=chat_states[0],
                            img_list=img_lists[0],
                            num_beams=num_beams,
                            temperature=temperature,
                            top_p=top_p,
                            max_new_tokens=512,
                            max_length=3000)[0]
    return response, chat_states[0], img_lists[0]


def extract_option_and_answer(text):
    # pattern = r'\(([A-Z])\)\s(.+)'
    # pattern = r'\((.*?)\) (.*?)\.'
    pattern = r'([A-E])\)\s*(.*?)\.'
    match = re.search(pattern, text)
    if match:
        option = match.group(1)
        answer = match.group(2)
        return f'({option}) {answer}.'
    else:
        return None
    
    
def main(args):
    num_beams = 1
    temperature = args.temperature
    top_p = args.top_p
    n_frms = args.num_frames
    eval_start_time = time.time()
    # prompt = "You are given a video associated with an audio. For the video and audio pair, answer the following question by picking the correct option."
    prompt = 'For the video and audio pair, choose the correct option. Do not provide any explanation or extra sentences.'


    # load model
    device = torch.device(f"cuda:{args.gpu_id}")
    args.options = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    # model_config.ckpt = args.video_llama_model_path
    if args.no_lora:
        model_config.lora = False

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()
    message = '\n' + '\n'.join([f'{k:<25}: {v}' for k, v in vars(args).items()])
    logging.info(message)

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    # load data
    video_path = args.video_path
    anno_path = args.anno_path
    anno_data = load_data(args, anno_path, split=args.split)

    vids = []
    vnames = []
    qids = []
    questions = []
    if args.sample_num > 0:
        # sample part data to evaluate
        anno_data = random.sample(anno_data, args.sample_num)
    for jterm in anno_data:
        vname = jterm["video"].split("/")[-1]
        
        # print('Hereeeee!')
        # if vname[-11:] in YT_INVALID_IDS:
        #     continue
        
        # if jterm['data_split'] != 'train':
        #     continue
        
        # if f'{vname}.mp4' not in COD_DOWNLOADED_VIDEOS:
        #     continue
        
        vid_path = os.path.join(video_path, vname)
        vids.append(vid_path)
        vnames.append(vname)
        qids.append(jterm["id"])
        choices = jterm["multi_choice"]
        
        if args.dataset == 'music_avqa' and jterm['is_binary'] == 'yes':
            choices = f" The options are: (A)yes, (B)no."
        else:
            choices = f" The options are: (A){choices[0]}, (B){choices[1]}, (C){choices[2]}, (D){choices[3]}."
            
        print(jterm["id"], choices)
        questions.append(jterm["question"] + choices)

    results = []
    bz = args.batch_size
    # evaluate using batch
    epoch = ceil(len(vnames) / bz)
    for i in tqdm(range(epoch)):
        sid = i * bz
        eid = min((i + 1) * bz, len(vnames))
        prompts = []
        # load video
        paths = vids[sid:eid]
        image_ids = qids[sid:eid]
        for pi in range(len(paths)):
            final_prompt = copy.deepcopy(prompt)
            idx = sid + pi
            prompts.append(final_prompt + " " + questions[idx])
        output, chat_state, img_lists = generate(chat, paths, prompts, num_beams, temperature, top_p, n_frms)
                
        results.append({
                "vname": vnames[sid],
                "raw_ans": output,
                "extracted_ans": extract_option_and_answer(output),
                "id": qids[sid],
                "prompt": chat_state.get_prompt()
            })

    # import pdb; pdb.set_trace()
    
    save_result(args, args.output_dir, results, args.split)

    # total_time = time.time() - eval_start_time
    # # convert seconds to date
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Evaluate time {}'.format(total_time_str))

    # with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #     f.write(json.dumps(cfg.to_dict(), indent=4) + "\n")
    #     f.write(message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='eval_configs/video_llama_eval_withaudio.yaml')
    parser.add_argument('--anno_path', type=str, default='')
    parser.add_argument('--video_path', type=str, default='')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--task', default='Absent_Answer_Detection')
    parser.add_argument('--dataset', default='AVQA')
    parser.add_argument('--output_dir', default='debug')
    parser.add_argument('--gt_file', default='', type=str)
    parser.add_argument('--split', default='val')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu_id', default='0')
    # parser.add_argument('--video_llama_model_path',
    #                     default='ckpt/video_llama_trained_model.pth')
    parser.add_argument('--sample_num', type=int, default=-1, help='fast inference by sampling N instances to evaluate')
    parser.add_argument('--example_output', action='store_true', help='output the example results')
    parser.add_argument('--no_lora', action='store_true')
    args = parser.parse_args()
    main(args)