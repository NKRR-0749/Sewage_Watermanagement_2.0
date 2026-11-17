"""
messages.py - How the agents talk to each other.

An agent sends a Message to another agent by name. Each agent reads its own
inbox during its turn and acts on what it finds. For example:

    MonitoringAgent  --ALERT-->        EqualizationAgent   (divert the shock!)
    ForecastAgent    --FORECAST-->     AerationAgent       (load coming next hour)
    AerationAgent    --TOXIC_SHOCK-->  ComplianceAgent     (expect bad effluent)
    ComplianceAgent  --RECIRCULATE-->  EqualizationAgent   (send failed water back)

Every message is also kept in a log so the dashboard can show the conversation.
"""

from dataclasses import dataclass, field


@dataclass
class Message:
    step: int
    sender: str
    receiver: str
    kind: str
    text: str                       # one line a human can read
    data: dict = field(default_factory=dict)

    def to_dict(self):
        return {"step": self.step, "sender": self.sender, "receiver": self.receiver,
                "kind": self.kind, "text": self.text}


class MessageBus:
    def __init__(self, keep_last=400):
        self.inboxes = {}
        self.log = []
        self.keep_last = keep_last

    def send(self, step, sender, receiver, kind, text, **data):
        msg = Message(step, sender, receiver, kind, text, data)
        self.inboxes.setdefault(receiver, []).append(msg)
        self.log.append(msg)
        if len(self.log) > self.keep_last:
            self.log = self.log[-self.keep_last:]
        return msg

    def read(self, receiver, kind=None):
        """Take (and remove) messages waiting for this receiver."""
        inbox = self.inboxes.get(receiver, [])
        taken = [m for m in inbox if kind is None or m.kind == kind]
        self.inboxes[receiver] = [m for m in inbox if m not in taken]
        return taken

    def messages_at(self, step):
        return [m for m in self.log if m.step == step]
