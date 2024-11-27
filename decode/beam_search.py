import torch
from model.utils import subsequent_mask


class Beam:

    def __init__(self, size, pad, bos, eos, device=False):
        self.size = size
        self._done = False
        self.PAD = pad
        self.BOS = bos
        self.EOS = eos
        # 每个翻译结果的分数
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # 每个时间步的回溯指针
        self.prev_ks = []

        # 每个时间步的输出
        # 初始化为 [BOS, PAD, PAD ..., PAD]
        self.next_ys = [torch.full(
            (size,), self.PAD, dtype=torch.long, device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_state(self):
        """ 获取当前时间步的输出序列 """
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """ 获取当前时间步的回溯指针 """
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_logprob):
        """ 更新 beam 状态，并判断是否结束 """
        num_words = word_logprob.size(1)

        # 累积之前的分数
        if len(self.prev_ks) > 0:
            beam_lk = word_logprob + \
                self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            # 初始化阶段
            beam_lk = word_logprob[0]

        # 展平以便找到 top-k 分数及其索引
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(
            self.size, 0, True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # 计算分数来自的 beam 和词汇
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # 如果 beam 顶部是 EOS，则标记为完成
        if self.next_ys[-1][0].item() == self.EOS:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """ 对分数进行排序 """
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """ 获取最高分及其索引 """
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """ 获取当前时间步的部分解码序列 """
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """ 回溯以构造完整的假设序列 """
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


def beam_search(model, src, src_mask, max_len, pad, bos, eos, beam_size, device):
    """ 使用 beam search 进行翻译 """

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        """ 映射实例索引到 tensor 位置 """
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        """ 收集仍处于活动状态的 tensor 部分 """
        _, *d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst * n_bm, *d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor

    def collate_active_info(src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list):
        """ 汇总活动实例信息以进行下一步解码 """
        n_prev_active_inst = len(inst_idx_to_position_map)
        active_inst_idx = [inst_idx_to_position_map[k]
                           for k in active_inst_idx_list]
        active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

        active_src_enc = collect_active_part(
            src_enc, active_inst_idx, n_prev_active_inst, beam_size)
        active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
            active_inst_idx_list)
        active_src_mask = collect_active_part(
            src_mask, active_inst_idx, n_prev_active_inst, beam_size)

        return active_src_enc, active_src_mask, active_inst_idx_to_position_map

    def beam_decode_step(inst_dec_beams, len_dec_seq, enc_output, inst_idx_to_position_map, n_bm):
        """ 解码并更新 beam 状态，返回活动的 beam 索引 """
        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state()
                               for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def predict_word(dec_seq, enc_output, n_active_inst, n_bm):
            out = model.decode(enc_output, src_mask,
                               dec_seq,
                               subsequent_mask(dec_seq.size(1))
                               .type_as(src.data))
            word_logprob = model.generator(out[:, -1])
            word_logprob = word_logprob.view(n_active_inst, n_bm, -1)

            return word_logprob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(
                    word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]

            return active_inst_idx_list

        n_active_inst = len(inst_idx_to_position_map)
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        word_logprob = predict_word(dec_seq, enc_output, n_active_inst, n_bm)
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_logprob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [inst_dec_beams[inst_idx].get_hypothesis(
                i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores

    with torch.no_grad():
        # 编码源句子
        src_enc = model.encode(src, src_mask)
        NBEST = beam_size
        batch_size, sent_len, h_dim = src_enc.size()

        # 扩展源编码和掩码以匹配 beam size
        src_enc = src_enc.repeat(1, beam_size, 1).view(
            batch_size * beam_size, sent_len, h_dim)
        src_mask = src_mask.repeat(1, beam_size, 1).view(
            batch_size * beam_size, 1, src_mask.shape[-1])

        inst_dec_beams = [Beam(beam_size, pad, bos, eos, device)
                          for _ in range(batch_size)]
        active_inst_idx_list = list(range(batch_size))
        inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(
            active_inst_idx_list)

        for len_dec_seq in range(1, max_len + 1):
            # 执行 beam 解码步骤
            active_inst_idx_list = beam_decode_step(
                inst_dec_beams, len_dec_seq, src_enc, inst_idx_to_position_map, beam_size)
            if not active_inst_idx_list:
                break

            # 汇总活动实例信息
            src_enc, src_mask, inst_idx_to_position_map = collate_active_info(
                src_enc, src_mask, inst_idx_to_position_map, active_inst_idx_list)

    batch_hyp, batch_scores = collect_hypothesis_and_scores(
        inst_dec_beams, NBEST)

    # 返回最高分的预测和分数
    batch_hyp = [hyp[0] for hyp in batch_hyp]
    batch_scores = [score[0] for score in batch_scores]

    return batch_hyp, batch_scores
