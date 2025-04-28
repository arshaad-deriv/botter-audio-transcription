import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Privacy Policy",
    page_icon="🔒",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.header {
    text-align: center;
    margin-bottom: 2rem;
}
.privacy-section {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='header'>Privacy Policy</h1>", unsafe_allow_html=True)

# Introduction
st.markdown("""
## Introduction
Thank you for using our Audio Transcription App. We are committed to protecting your privacy and ensuring the security of your data.
This privacy policy explains how we collect, use, and safeguard your information when you use our service.
""")

# Information Collection section
st.markdown("""
<div class='privacy-section'>
<h2>Information We Collect</h2>

<h3>Audio Data</h3>
<ul>
<li>We temporarily process audio recordings you provide for transcription purposes</li>
<li>Audio files are processed in memory and not permanently stored on our servers</li>
<li>Transcriptions are displayed to you but not retained after your session ends unless you explicitly save them</li>
</ul>

<h3>API Keys</h3>
<ul>
<li>OpenAI and Claude API keys you provide are stored in your browser session</li>
<li>We never store your API keys on our servers</li>
<li>Your keys are used solely to access transcription and AI services on your behalf</li>
</ul>

<h3>Google Calendar Data (Optional)</h3>
<ul>
<li>If you choose to use the Google Calendar integration, we access your calendar data only with your explicit permission</li>
<li>Calendar access is used solely to associate transcripts with your meetings</li>
<li>We do not store your Google credentials on our servers</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Data Usage section
st.markdown("""
<div class='privacy-section'>
<h2>How We Use Your Information</h2>

<ul>
<li>To provide transcription and summarization services</li>
<li>To improve the functionality and user experience of our application</li>
<li>We do not sell or share your data with third parties</li>
<li>We do not use your data for advertising purposes</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Data Security section
st.markdown("""
<div class='privacy-section'>
<h2>Data Security</h2>

<ul>
<li>All data is transmitted using secure HTTPS connections</li>
<li>Audio processing is performed using trusted third-party services (OpenAI, Anthropic)</li>
<li>Files are processed temporarily and not stored permanently</li>
<li>We implement reasonable security measures to protect your information</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Third-party Services section
st.markdown("""
<div class='privacy-section'>
<h2>Third-party Services</h2>

<p>Our app utilizes the following third-party services:</p>

<ul>
<li><strong>OpenAI</strong> - Used for audio transcription and summarization</li>
<li><strong>Anthropic</strong> - Used for Claude AI summarization (optional)</li>
<li><strong>Google</strong> - Used for calendar integration (optional)</li>
<li><strong>Streamlit</strong> - Application hosting platform</li>
</ul>

<p>Each of these services has their own privacy policies that apply to your data when processed by them.</p>
</div>
""", unsafe_allow_html=True)

# User Rights section
st.markdown("""
<div class='privacy-section'>
<h2>Your Rights</h2>

<p>You have the right to:</p>

<ul>
<li>Access your data</li>
<li>Delete your data</li>
<li>Withdraw consent for data processing</li>
<li>Opt out of any future communications</li>
</ul>

<p>To exercise these rights, please contact us using the information below.</p>
</div>
""", unsafe_allow_html=True)

# Contact Information
st.markdown("""
<div class='privacy-section'>
<h2>Contact Us</h2>

<p>If you have any questions or concerns about our privacy policy, please contact us at:</p>

<p>Email: arshaad.mohiadden@deriv.com</p>
</div>
""", unsafe_allow_html=True)

# Return button to main app
if st.button("Return to Main App", key="return_button", type="primary"):
    # Redirect to the main app
    import webbrowser
    webbrowser.open_new_tab("/")
