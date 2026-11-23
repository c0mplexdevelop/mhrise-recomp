# PC players in Local Play (crossplay)

MH Rise on PC (Steam) and on the Switch send the same game messages: the PC version only carries them over Steam's
lobbies and messages instead of Pia (the Switch's network library) over Local Play. So a PC player can join a hub this
port hosts through a bridge on the PC: a `dinput8.dll` next to the PC game ([rise-pc-netlog](https://github.com/klark231/rise-pc-netlog)
has the research and the bridge, `rise-bridge/` and `BRIDGE.md`; with REFramework installed it goes in as the plugin
`reframework/plugins/rise_crossplay.dll` instead) that answers the game's Steam lobby and message calls itself,
reaches the port's hubs through the same hub servers and link protocol as a phone, and speaks Pia for its player.

What the port does for it (`runtime/src/nnsdk/ldn.cpp`, `pia_key.h`):

- A station whose HELLO has kind 2 (where a search has 1) is a bridge. The host answers it like any other station,
  then sends a KEY frame with Pia's key for the network, and again whenever Pia starts a new session with a new key.
  Pia encrypts every packet with it (AES-128-GCM), and the bridge needs it to be one of the network's stations; the
  host has it because Pia expands it in this process (its key schedule is hooked, 16.0.1 only). rise-bridge's
  `hublink.py` speaks this (it called it netlog3.patch); it joins with link version 1, which the host still takes.
- Everything else is as for a phone: the bridge's Pia packets go through the host like any station's.

Only a hub the port hosts takes PC players for now; a PC player hosting phones would need the bridge to be Pia's host.
`<data>/no_pc_players` (mhr.ini) turns it off: no hook, and a bridge's hello gets no key.

## Where it stands (Nov 24)

- **Works:** the PC finds the port's hub, joins, both stay in; the PC shows the port player's hunter and sees it walk;
  guild cards go both ways.
- **The port doesn't show the PC's hunter.** The walking messages arrive, but the hunter's periodic state from the PC is
  packed differently (its first message ~100 bytes shorter than any Switch one), and the Switch game never puts that
  hunter in. Details in rise-pc-netlog's `analysis/NOTES.md`, section 9.
- **Quests:** the PC posts a quest as a separate Steam lobby, the Switch inside the same room. The bridge doesn't
  translate that yet, so the PC can't join the port's quest and the port gets stuck loading into the PC's.

To see where the port's game stops with the PC hunter, rise-pc-netlog's `probe/crossplay-probe.zip` goes in through
Settings → Mods & cheats → Add a mod (this branch's build has the Lua runtime). It logs `[xprobe]` lines to `log.txt`:
the player slots, the hunter-creating calls, and the hunter packets that arrive. The same script runs on the PC with
REFramework. It only reads.
