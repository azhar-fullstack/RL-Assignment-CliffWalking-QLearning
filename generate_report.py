"""Generate a student-style technical report PDF (max 4 pages) with title page."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSETS = ROOT / "report" / "title_assets"
OUT = ROOT / "report" / "Technical_Report.pdf"

# Student details (from Uni/title page.docx)
STUDENT_NAME = "Azhar Mehmood"
REG_NO = "FA23-BAI-039"
COURSE = "Reinforcement Learning / AI Assignment"
ASSIGNMENT_TITLE = (
    "Custom Reinforcement Learning Agent Implementation using Farama Gymnasium"
)


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="UniTitle",
            fontName="Times-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4472C4"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverLine",
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4472C4"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverBody",
            fontName="Times-Roman",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverMeta",
            fontName="Times-Bold",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0D0D0D"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubCenter",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2Custom",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=11.5,
            spaceBefore=9,
            spaceAfter=4,
            textColor=colors.HexColor("#1F4E79"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H3Custom",
            parent=styles["Heading3"],
            fontName="Times-Bold",
            fontSize=10.5,
            spaceBefore=6,
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontName="Times-Italic",
            fontSize=8.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Formula",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=10,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    return styles


def maybe_image(path: Path, width: float, height: float):
    if path.exists():
        return Image(str(path), width=width, height=height)
    return Paragraph(f"[Missing figure: {path.name}]", getSampleStyleSheet()["Normal"])


def build_title_page(styles):
    """Simple cover modeled on Uni/title page.docx."""
    story = []
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("ASSIGNMENT", styles["UniTitle"]))
    story.append(Spacer(1, 0.15 * inch))

    logo = ASSETS / "image1.png"
    if logo.exists():
        img = Image(str(logo), width=1.7 * inch, height=1.55 * inch)
        img.hAlign = "CENTER"
        story.append(img)
    story.append(Spacer(1, 0.35 * inch))

    story.append(Paragraph(ASSIGNMENT_TITLE, styles["CoverLine"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(COURSE, styles["CoverBody"]))
    story.append(
        Paragraph(
            "Environment: CliffWalking-v1 &nbsp;&nbsp;|&nbsp;&nbsp; Algorithm: Q-Learning",
            styles["CoverBody"],
        )
    )
    story.append(Spacer(1, 0.7 * inch))
    story.append(Paragraph(f"Name: {STUDENT_NAME}", styles["CoverMeta"]))
    story.append(Paragraph(f"Reg. No: {REG_NO}", styles["CoverMeta"]))
    story.append(Spacer(1, 0.35 * inch))
    story.append(Paragraph("Submission: GitHub Repository Link + Technical Report (PDF)", styles["CoverBody"]))
    story.append(Paragraph("Due Date: 20/10/2026", styles["CoverBody"]))
    story.append(PageBreak())
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title="Technical Report - CliffWalking Q-Learning",
        author=STUDENT_NAME,
    )

    story = []
    story.extend(build_title_page(styles))

    story.append(Paragraph("Technical Report", styles["TitleCenter"]))
    story.append(
        Paragraph(
            f"{STUDENT_NAME} ({REG_NO}) &nbsp;|&nbsp; CliffWalking-v1 &nbsp;|&nbsp; Q-Learning from scratch",
            styles["SubCenter"],
        )
    )

    # 1. Problem Description
    story.append(Paragraph("1. Problem Description", styles["H2Custom"]))
    story.append(
        Paragraph(
            "For this assignment I picked <b>CliffWalking-v1</b> from Gymnasium. It is a small 4x12 grid "
            "(48 states), so I could implement the algorithm myself without needing neural networks. "
            "The agent starts at the bottom-left cell and has to reach the goal at the bottom-right. "
            "There is a cliff along the bottom row. If the agent steps into the cliff, it gets a big "
            "penalty and goes back to the start. Actions are simple: "
            "<b>0 = Up, 1 = Right, 2 = Down, 3 = Left</b>.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "Reward setup is also straightforward. Every step gives <b>-1</b>. Falling into the cliff "
            "gives <b>-100</b>. Reaching the goal ends the episode. The best safe path is 13 steps, so "
            "the best score is <b>-13</b>. If the agent takes a longer upper path, the score gets worse "
            "(more negative). That tradeoff is why this environment is useful for testing an RL agent.",
            styles["BodyJust"],
        )
    )

    # 2. Methodology
    story.append(Paragraph("2. Methodology", styles["H2Custom"]))
    story.append(Paragraph("2.1 Algorithm", styles["H3Custom"]))
    story.append(
        Paragraph(
            "I implemented <b>Q-Learning</b> with an epsilon-greedy policy. The idea is to keep a table "
            "Q(s, a) for every state-action pair. Most of the time the agent picks the action with the "
            "highest Q value, but with probability epsilon it picks a random action so it can explore. "
            "After each step, the table is updated with:",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "Q(s,a) = Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ]",
            styles["Formula"],
        )
    )
    story.append(
        Paragraph(
            "Here alpha is the learning rate and gamma is the discount factor. Epsilon starts at 1.0 and "
            "slowly decays after every episode down to 0.01. I used the max over next actions in the "
            "update, which is the normal off-policy Q-Learning rule.",
            styles["BodyJust"],
        )
    )
    story.append(Paragraph("2.2 Code Structure", styles["H3Custom"]))
    story.append(
        Paragraph(
            "I split the code into three files so it stays easy to read. <b>agent.py</b> has the "
            "QLearningAgent class (choose action, update Q, decay epsilon, save/load). "
            "<b>utils.py</b> has training, evaluation, plots, and GIF rendering. <b>main.py</b> runs "
            "the full pipeline from the terminal. The environment is created with "
            "<i>gymnasium.make(\"CliffWalking-v1\")</i>. After training, I save the Q-table as "
            "<i>models/q_table.npy</i> and a short demo clip as <i>results/agent_demo.gif</i>.",
            styles["BodyJust"],
        )
    )

    # 3. Experimental Setup
    story.append(Paragraph("3. Experimental Setup and Hyperparameters", styles["H2Custom"]))
    story.append(
        Paragraph(
            "I trained in Python using Gymnasium. The table below shows the final settings I kept for "
            "the saved model and GIF. For the sensitivity test, I trained again for 3000 episodes and "
            "changed one setting at a time (same seed).",
            styles["BodyJust"],
        )
    )

    table_data = [
        ["Hyperparameter", "Symbol", "Value", "Why I chose it"],
        ["Learning rate", "alpha", "0.5", "0.1 was slow; 0.9 was less stable"],
        ["Discount factor", "gamma", "0.99", "Helps the agent care about reaching goal"],
        ["Start epsilon", "eps0", "1.0", "Start fully random"],
        ["Min epsilon", "eps_min", "0.01", "Keep a little exploration"],
        ["Epsilon decay", "eps_decay", "0.995", "Decay once per episode"],
        ["Training episodes", "N", "5000", "Main training run"],
        ["Max steps", "T", "200", "Stop very long episodes"],
        ["Eval episodes", "-", "100", "Test greedy policy"],
        ["Seed", "-", "42", "Same seed for fair comparison"],
    ]
    table = Table(table_data, colWidths=[1.45 * inch, 0.75 * inch, 0.75 * inch, 2.85 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.Color(0.93, 0.93, 0.97)]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 6))

    # 4. Results
    story.append(Paragraph("4. Results and Discussion", styles["H2Custom"]))
    story.append(
        Paragraph(
            "After 5000 episodes, I tested the greedy policy for 100 episodes. The average return was "
            "<b>-13.0</b> with <b>0</b> variance, and success rate was <b>100%</b>. That means the agent "
            "learned the optimal 13-step path. Looking at Figure 1, reward improves a lot in the first "
            "1000-2000 episodes, then becomes more stable as epsilon gets small. Figure 2 shows the "
            "learned values on the grid. Cells near the good path have higher values, and cells near "
            "the cliff look worse, which makes sense.",
            styles["BodyJust"],
        )
    )
    story.append(maybe_image(RESULTS / "training_curves.png", width=6.2 * inch, height=1.9 * inch))
    story.append(Paragraph("Figure 1. Reward, success rate, and epsilon during training.", styles["Caption"]))

    story.append(maybe_image(RESULTS / "q_value_heatmap.png", width=5.9 * inch, height=1.65 * inch))
    story.append(Paragraph("Figure 2. Max Q-value for each cell on the 4x12 map.", styles["Caption"]))

    story.append(PageBreak())
    story.append(
        Paragraph(
            "For hyperparameter checks (Figure 3), alpha = 0.5 worked better than alpha = 0.1 in the "
            "same episode budget. Gamma = 0.99 was fine for this task. When I decayed epsilon too slowly "
            "(0.999), the agent kept exploring for longer, so the late training return stayed worse. "
            "Overall, learning rate and epsilon decay affected the results the most in my runs.",
            styles["BodyJust"],
        )
    )
    story.append(
        maybe_image(RESULTS / "hyperparameter_sensitivity.png", width=5.7 * inch, height=2.55 * inch)
    )
    story.append(
        Paragraph(
            "Figure 3. Average return in the last 500 episodes for different settings.",
            styles["Caption"],
        )
    )

    story.append(Paragraph("4.1 Problems I Noticed", styles["H3Custom"]))
    story.append(
        Paragraph(
            "Early in training the agent falls into the cliff a lot because epsilon is high. That looks "
            "bad in the reward plot, but it is normal while it is still exploring. Sometimes the agent "
            "can also learn a safer but longer path on the upper rows, which gives a worse score than "
            "-13. Another limit is that this is a table method, so it works for CliffWalking, but it "
            "would not directly work for continuous games like CartPole unless I switch to something "
            "like DQN. The demo GIF shows the final greedy policy walking from start to goal without "
            "falling.",
            styles["BodyJust"],
        )
    )

    # 5. Conclusion
    story.append(Paragraph("5. Conclusion", styles["H2Custom"]))
    story.append(
        Paragraph(
            "In this assignment I built a Q-Learning agent from scratch on CliffWalking-v1, trained it, "
            "plotted the learning curves, saved the Q-table, and recorded a short GIF of the final "
            "policy. The agent reached the optimal score (-13) in evaluation. Changing alpha, gamma, and "
            "epsilon decay clearly changed how fast and how well it learned. If I continue this work, I "
            "would try SARSA (usually safer near the cliff) or Double Q-Learning next.",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "<b>GitHub:</b> https://github.com/azhar-fullstack/RL-Assignment-CliffWalking-QLearning<br/>"
            "<b>Main files:</b> main.py, agent.py, utils.py, notebooks/training.ipynb, models/q_table.npy, "
            "results/agent_demo.gif",
            styles["BodyJust"],
        )
    )

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
