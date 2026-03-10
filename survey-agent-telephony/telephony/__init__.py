# telephony package — VICIdial + Jambonz integration for survey voice agent
#
# Architecture:
#   VICIdial (dialer) → SIP transfer → Jambonz (SIP↔WS bridge) → FastAPI → VoiceAgent → Gemini
#   After survey → VICIdial API (disposition, lead update, hangup)
