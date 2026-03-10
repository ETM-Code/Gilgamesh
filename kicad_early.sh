#!/bin/bash

# Colors
RED='\033[1;31m'
YEL='\033[1;33m'
GRN='\033[1;32m'
CYN='\033[1;36m'
WHT='\033[1;37m'
DIM='\033[2m'
RST='\033[0m'

clear

echo ""
echo -e "${CYN}╭─────────────────────────────────────────────────────────────╮${RST}"
echo -e "${CYN}│${RST}  ${WHT}sys${DIM} // $(hostname)${RST}$(printf '%*s' $((46 - ${#HOSTNAME})) '')${CYN}│${RST}"
echo -e "${CYN}╰─────────────────────────────────────────────────────────────╯${RST}"
echo ""
echo -e "  ${DIM}uptime${RST}     4h 38m"
echo -e "  ${DIM}procs${RST}      412"
echo ""
echo -e "  ${DIM}cpu${RST}        ${GRN}14%${RST}     ${GRN}█████${DIM}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░${RST}"
echo -e "  ${DIM}mem${RST}        ${GRN}6.2 GB ${DIM}/ 16 GB${RST}"
echo -e "             ${GRN}█████████████${DIM}░░░░░░░░░░░░░░░░░░░░░░░${RST}"
echo -e "  ${DIM}swap${RST}       ${GRN}0.0 GB ${DIM}/ 4 GB${RST}"
echo -e "             ${DIM}░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░${RST}"
echo ""
echo -e "  ${DIM}PID      MEM          CPU     PROCESS${RST}"
echo -e "  ${DIM}─────    ───          ───     ───────${RST}"
echo -e "  ${WHT}18204    3.1 GB       6%      kicad ${DIM}(pcbnew)${RST}"
echo -e "  ${DIM}1847     1.2 GB       4%      Google Chrome ${DIM}(Helper)(GPU)${RST}"
echo -e "  ${DIM}1291     480 MB       2%      Code ${DIM}(Visual Studio Code)${RST}"
echo -e "  ${DIM}412      384 MB       3%      WindowServer${RST}"
echo -e "  ${DIM}1844     310 MB       1%      Google Chrome${RST}"
echo -e "  ${DIM}1102     142 MB       0%      Safari${RST}"
echo -e "  ${DIM}507      118 MB       0%      mds_stores${RST}"
echo -e "  ${DIM}1906      86 MB       0%      TextEdit${RST}"
echo -e "  ${DIM}298       74 MB       0%      coreaudiod${RST}"
echo -e "  ${DIM}184       61 MB       0%      loginwindow${RST}"
echo -e "  ${DIM}391       58 MB       0%      Finder${RST}"
echo -e "  ${DIM}203       47 MB       0%      cfprefsd${RST}"
echo -e "  ${DIM}518       38 MB       0%      sharingd${RST}"
echo -e "  ${DIM}294       34 MB       0%      bluetoothd${RST}"
echo -e "  ${DIM}176       31 MB       0%      distnoted${RST}"