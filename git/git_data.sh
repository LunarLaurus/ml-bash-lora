#!/bin/bash

# Mainline handheld Pokémon (Gen I–IV) — pret .git URLs
poke_repos_mainline=(
    "https://github.com/pret/pokered.git"        # Red/Blue (GB) - Gen I
    "https://github.com/pret/pokeyellow.git"     # Yellow (GB) - Gen I
    "https://github.com/pret/pokegold.git"       # Gold/Silver (GBC) - Gen II
    "https://github.com/pret/pokecrystal.git"    # Crystal (GBC) - Gen II
    "https://github.com/pret/pokefirered.git"    # FireRed/LeafGreen (GBA) - Gen III remakes
    "https://github.com/pret/pokeruby.git"       # Ruby/Sapphire (GBA) - Gen III
    "https://github.com/pret/pokeemerald.git"    # Emerald (GBA) - Gen III
    "https://github.com/pret/pokediamond.git"    # Diamond/Pearl (NDS) - Gen IV
    "https://github.com/pret/pokeplatinum.git"   # Platinum (NDS) - Gen IV
    "https://github.com/pret/pokeheartgold.git"  # HeartGold/SoulSilver (NDS) - Gen IV remakes
)

BUILD_DIR="$SCRIPT_DIR/build"
TS_SO="$BUILD_DIR/my-languages.so"
TS_C_REPO="$SCRIPT_DIR/tree-sitter-c"
TS_ASM_REPO="$SCRIPT_DIR/tree-sitter-asm"
TS_BIN="$BUILD_DIR/tree-sitter"