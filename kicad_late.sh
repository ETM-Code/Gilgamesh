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
echo -e "  ${DIM}uptime${RST}     5h 11m"
echo -e "  ${DIM}procs${RST}      408"
echo ""
echo -e "  ${DIM}cpu${RST}        ${RED}96%${RST}     ${RED}██████████████████████████████████${DIM}░${RST}"
echo -e "  ${DIM}mem${RST}        ${RED}15.4 GB ${DIM}/ 16 GB${RST}"
echo -e "             ${RED}██████████████████████████████████${DIM}░░${RST}"
echo -e "  ${DIM}swap${RST}       ${YEL}3.2 GB ${DIM}/ 4 GB${RST}"
echo -e "             ${YEL}████████████████████████████${DIM}░░░░░░░░${RST}"
echo ""
echo -e "  ${DIM}PID      MEM          CPU     PROCESS${RST}"
echo -e "  ${DIM}─────    ───          ───     ───────${RST}"
echo -e "  ${RED}18204    13.8 GB      91%     kicad ${DIM}(pcbnew)${RST}"
echo -e "  ${DIM}1847     1.3 GB        3%     Google Chrome ${DIM}(Helper)(GPU)${RST}"
echo -e "  ${DIM}1291     512 MB        1%     Code ${DIM}(Visual Studio Code)${RST}"
echo -e "  ${DIM}412      397 MB        2%     WindowServer${RST}"
echo -e "  ${DIM}1844     318 MB        1%     Google Chrome${RST}"
echo -e "  ${DIM}1102     148 MB        0%     Safari${RST}"
echo -e "  ${DIM}507      122 MB        0%     mds_stores${RST}"
echo -e "  ${DIM}1906      86 MB        0%     TextEdit${RST}"
echo -e "  ${DIM}298       71 MB        0%     coreaudiod${RST}"
echo -e "  ${DIM}184       63 MB        0%     loginwindow${RST}"
echo -e "  ${DIM}391       55 MB        0%     Finder${RST}"
echo -e "  ${DIM}203       49 MB        0%     cfprefsd${RST}"
echo -e "  ${DIM}518       38 MB        0%     sharingd${RST}"
echo -e "  ${DIM}294       34 MB        0%     bluetoothd${RST}"
echo -e "  ${DIM}176       28 MB        0%     distnoted${RST}"