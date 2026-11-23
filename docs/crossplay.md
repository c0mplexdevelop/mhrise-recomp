# PC players in Local Play (crossplay)

MH Rise on PC (Steam) and on the Switch send the same game messages: the PC version only carries them over Steam's
lobbies and messages instead of Pia (the Switch's network library) over Local Play. So a PC player can join a hub this
port hosts through a bridge on the PC: a `dinput8.dll` next to the PC game ([rise-pc-netlog](https://github.com/klark231/rise-pc-netlog)
has the research it's built on, in `analysis/NOTES.md`) that answers the game's Steam lobby and message calls itself,
reaches the port's hubs through the same hub servers and link protocol as a phone, and speaks Pia for its player.

What the port does for it (`runtime/src/nnsdk/ldn.cpp`, `pia_key.h`):

- A station whose HELLO ends in a flags byte with bit 0 set is a bridge. The host answers it like any other station,
  plus Pia's key for the network (16 bytes after the UDP token in the WELCOME), and a KEY frame whenever Pia starts
  a new session with a new key. Pia encrypts every packet with it (AES-128-GCM), and the bridge needs it to be one of
  the network's stations; the host has it because Pia expands it in this process (its key schedule is hooked, 16.0.1
  only).
- Everything else is as for a phone: the bridge's Pia packets go through the host like any station's.

Only a hub the port hosts takes PC players for now; a PC player hosting phones would need the bridge to be Pia's host.
`<data>/no_pc_players` (mhr.ini) turns it off: no hook, and a bridge's hello gets no key.
