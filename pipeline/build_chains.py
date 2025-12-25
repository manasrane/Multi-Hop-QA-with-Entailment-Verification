from itertools import combinations

def build_chains(passages, max_chain_length=2):
    """Build chains of passages up to max_chain_length"""
    chains = []
    for length in range(1, max_chain_length + 1):
        if length == 1:
            chains.extend([[p] for p in passages])
        else:
            combos = list(combinations(passages, length))
            chains.extend([list(combo) for combo in combos])
    return chains

def build_chains_efficient(passages, max_chain_length=2):
    """More efficient chain building"""
    chains = []
    for length in range(1, max_chain_length + 1):
        if length == 1:
            chains.extend(passages)
        else:
            combos = list(combinations(range(len(passages)), length))
            for combo in combos:
                chain = [passages[i] for i in combo]
                chains.append(chain)
    return chains