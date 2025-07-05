
FROM ghcr.io/prefix-dev/pixi:0.49.0 AS build

# Get updates and security fixes as root user before switching to non-root user
#   https://pythonspeed.com/articles/security-updates-in-docker/
#   The worst so-called “best practice” for Docker
#   by Itamar Turner-Trauring
#   Last updated 01 Oct 2021, originally created 23 Mar 2021
RUN apt-get update && apt-get -y upgrade


# Switch to non-root user
RUN useradd --create-home appuser
WORKDIR /app
COPY . . 

# Change ownership of /app to appuser
RUN chown -R appuser:appuser /app
USER appuser


RUN pixi install --locked --environment prod

EXPOSE 8050
ENTRYPOINT ["pixi", "run", "--"]
CMD ["python", "-m", "src.app.app"]
