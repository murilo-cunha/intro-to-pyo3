FROM python:3.12.0-slim-bookworm
# Update default packages
RUN apt-get -qq update

# Get required packages
RUN apt-get install -y -q \
    build-essential \
    curl

# Install Rust and make it available in the PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Check that Rust is installed
CMD ["cargo", "--version"]
