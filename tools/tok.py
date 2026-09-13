"""SentencePiece-BPE tokenizer over the blob embedded in the `.cact`."""

SP_SPACE = "▁"

TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4


class Tokenizer:
    def __init__(self, meta):
        self.pieces = meta["pieces"]
        self.scores = meta["scores"]
        self.types = meta["types"]
        self.add_dummy = meta["add_dummy_prefix"]
        self.byte_fallback = meta["byte_fallback"]
        self.unk_id = meta["unk_id"]
        self.eos_id = meta["eos_id"]
        self.bos_id = meta["bos_id"]
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i for i, (p, t)
                        in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types)
                               if t == TK_USER_DEFINED), key=len, reverse=True)

    def _bpe(self, seg):
        syms = list(seg)
        while len(syms) > 1:
            best, best_j = None, -1
            for j in range(len(syms) - 1):
                i = self.p2id.get(syms[j] + syms[j + 1])
                if i is not None and (best is None or self.scores[i] > best):
                    best, best_j = self.scores[i], j
            if best_j < 0:
                break
            syms[best_j:best_j + 2] = [syms[best_j] + syms[best_j + 1]]
        ids = []
        for s in syms:
            i = self.p2id.get(s)
            if i is not None:
                ids.append(i)
            elif self.byte_fallback:
                ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append(self.unk_id)
        return ids

    def encode(self, text):
        if not text:
            return []
        esc = text.replace(" ", SP_SPACE)
        if self.add_dummy:
            esc = SP_SPACE + esc
        ids, buf, i, n = [], [], 0, len(esc)
        while i < n:
            marker = next((m for m in self.markers if esc.startswith(m, i)), None)
            if marker is not None:
                ids += self._bpe("".join(buf))
                buf = []
                ids.append(self.p2id[marker])
                i += len(marker)
            else:
                buf.append(esc[i])
                i += 1
        ids += self._bpe("".join(buf))
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(SP_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text

    def piece(self, i):
        return self.pieces[i]
