FROM grafana/grafana:latest

# # Copy your custom Grafana configuration and dashboards
# COPY ./dashboards /var/lib/grafana/dashboards
# COPY ./grafana.ini /etc/grafana/grafana.ini

# Ensure dashboards directory is writable by Grafana user
# RUN chmod -R 777 /var/lib/grafana/dashboards

# Expose Grafana's default port
EXPOSE 3000

# Run Grafana
CMD ["grafana-server", "--homepath=/usr/share/grafana"]